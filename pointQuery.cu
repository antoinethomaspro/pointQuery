#include <optix.h>
#include "pointQuery.h"

extern "C" {
__constant__ Params params;
}



extern "C" __global__ void __closesthit__radiance()
{ /*! for this simple example, this will remain empty */ }

extern "C" __global__ void __anyhit__radiance()
{ /*! for this simple example, this will remain empty */ }



//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__radiance()
{ /*! for this simple example, this will remain empty */ }



//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame()
{
if (params.frameID == 0 &&
    optixGetLaunchIndex().x == 0 &&
    optixGetLaunchIndex().y == 0) {
    // we could of course also have used optixGetLaunchDims to query
    // the launch size, but accessing the optixLaunchParams here
    // makes sure they're not getting optimized away (because
    // otherwise they'd not get used)
    printf("############################################\n");
    printf("Hello world from OptiX 7 raygen program!\n(within a %ix%i-sized launch)\n",
            params.fbSize.x,
            params.fbSize.y);
    printf("############################################\n");
}