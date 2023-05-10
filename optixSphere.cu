//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>

#include "optixSphere.h"
#include <cuda/helpers.h>

#include <sutil/vec_math.h>

// #include "sphere.h"


#define float3_as_ints( u ) float_as_int( u.x ), float_as_int( u.y ), float_as_int( u.z )

extern "C" {
__constant__ Params params;
}

/*
Tetrahedron:                          Tetrahedron10:

                   v
                 .
               ,/
              /
           3                                     2
         ,/|`\                                 ,/|`\
       ,/  |  `\                             ,/  |  `\
     ,/    '.   `\                         ,6    '.   `5
   ,/       |     `\                     ,/       8     `\
 ,/         |       `\                 ,/         |       `\
1-----------'.--------2 --> u         0--------4--'.--------1
 `\.         |      ,/                 `\.         |      ,/
    `\.      |    ,/                      `\.      |    ,9
       `\.   '. ,/                           `7.   '. ,/
          `\. |/                                `\. |/
             `4                                    `3
                `\.
                   ` w

*/

extern "C" __global__ void __intersection__behind__triangle()
{
    const TetrahedronIndex* hit_group_data = reinterpret_cast<TetrahedronIndex*>( optixGetSbtDataPointer() );
    const int primID = optixGetPrimitiveIndex();
    const float3 ray_origin = optixGetWorldRayOrigin();

    const int index1 = hit_group_data->indices[primID * 4];
    const int index2 = hit_group_data->indices[primID * 4 + 1];
    const int index3 = hit_group_data->indices[primID * 4 + 2];
    const int index4 = hit_group_data->indices[primID * 4 + 3];
    
    const float3 v1 = hit_group_data->vertices[index1];
    const float3 v2 = hit_group_data->vertices[index2];
    const float3 v3 = hit_group_data->vertices[index3];
    const float3 v4 = hit_group_data->vertices[index4];

    const float3 norm1 = normalize(cross((v4-v1),(v3-v1)));
    const float3 norm2 = normalize(cross((v2-v4),(v3-v4)));
    const float3 norm3 = normalize(cross((v2-v1),(v4-v1)));
    const float3 norm4 = normalize(cross((v3-v1),(v2-v1)));

    if((dot(norm1, (ray_origin - v1)) < 1e-5) && 
       (dot(norm2, (ray_origin - v4)) < 1e-5) &&
       (dot(norm3, (ray_origin - v1)) < 1e-5) && 
       (dot(norm4, (ray_origin - v1)) < 1e-5)){          
            optixReportIntersection(0., 0., primID);
    }
}

static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        float3*                prd
        )
{
    unsigned int p0, p1, p2;
    p0 = float_as_int( prd->x );
    p1 = float_as_int( prd->y );
    p2 = float_as_int( prd->z );
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset
            0,                   // SBT stride
            0,                   // missSBTIndex
            p0, p1, p2 );
    prd->x = int_as_float( p0 );
    prd->y = int_as_float( p1 );
    prd->z = int_as_float( p2 );
}

static __forceinline__ __device__ void setPayload( float3 p )
{
    optixSetPayload_0( float_as_int( p.x ) );
    optixSetPayload_1( float_as_int( p.y ) );
    optixSetPayload_2( float_as_int( p.z ) );
}

static __forceinline__ __device__ float3 getPayload()
{
    return make_float3(
            int_as_float( optixGetPayload_0() ),
            int_as_float( optixGetPayload_1() ),
            int_as_float( optixGetPayload_2() )
            );
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();
    const float3      U      = rtData->camera_u;
    const float3      V      = rtData->camera_v;
    const float3      W      = rtData->camera_w;
    const float2      d = 2.0f * make_float2(
            static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
            static_cast<float>( idx.y ) / static_cast<float>( dim.y )
            ) - 1.0f;

    float3 originPrimaryRay      = rtData->cam_eye;
    const float3 direction   = normalize( d.x * U + d.y * V + W );
    float3 payload_rgb = make_float3( 0.0f, 0.0f, 0.0f );

    for (float i = 0.f; i < 10.f; i+=0.05f ) {

        float3 origin = originPrimaryRay + direction*i;

        trace( params.handle,
            origin,
            direction,
            0.00f,  // tmin
            1e-16f,  // tmax
            &payload_rgb );


    // Code to be executed in each iteration of the loop goes here
    
    }

    params.image[idx.y * params.image_width + idx.x] = make_color( payload_rgb );
}


extern "C" __global__ void __miss__ms()
{

}


extern "C" __global__ void __closesthit__ch()
{
    float3    payload = getPayload();
    int primID = optixGetAttribute_0();

    switch(primID){
        case 0:
            setPayload( payload + make_float3( 0.05f, 0.f, 0.f));
            break;
        case 1:
            setPayload( payload + make_float3( 0.f, 0.05f, 0.f));
            break;
        case 2:
            setPayload( payload + make_float3( 0.f, 0.f, 0.05f));
            break;
        case 3:
            setPayload( payload + make_float3( 0.05, 0.05f, 0.f));
            break;
        case 4:
            setPayload( payload + make_float3( 0.f, 0.05f, 0.05f));
            break;
    }
    
}
