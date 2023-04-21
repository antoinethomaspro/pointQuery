// <<glad/glad.h> must be included before gl_interop

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

// Include the automatically generated header file from sampleConfig.h
// An environment variable is defined, such as a path to a directory
#include <sampleConfig.h>

// Include header files provided by OptiX SDK
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <optix_stack_size.h>

#include <GLFW/glfw3.h>

#include "optixInOneWeekend.h"
#include "random.h"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

bool resize_dirty   = false;
bool minimized      = false;

// camera
bool camera_changed = true;
sutil::Camera    camera;
sutil::Trackball trackball;

// mouse
int32_t mouse_button = -1;

// Number of samples per pixel per kernel launch
int32_t samples_per_launch = 16;

// A shader record that makes up a shader binding table and consists of a header and arbitrary data.
// The header size is fixed at OPTIX_SBT_RECORD_HEADER_SIZE (32 bytes) in OptiX 7.4.
// Data can store user-defined data types. However, in the Shader binding table
// Each HitGroup record, Miss record, and Callables record can hold multiple records.
// Record sizes must be equal.

template <typename T>
struct Record 
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RayGenRecord   = Record<RayGenData>;
using MissRecord     = Record<MissData>;
using HitGroupRecord = Record<HitGroupData>;
using EmptyRecord    = Record<EmptyData>;

// To call a Direct/Continuation callable program on the device (GPU) side,
// Since the SBT_ID of the Callables program is required, assign a number in the order of generation and wake up,
// Build SBT for Callables in that order
struct CallableProgram
{
    OptixProgramGroup program = nullptr;
    uint32_t          id      = 0;
};

// Geometry acceleration structure (GAS) 
// When linking the traversable handle of GAS to OptixInstance,
// Knowing the number of SBT records held by GAS,
// Easy to build sbt offset of Instance all at once
struct GeometryAccelData
{
    OptixTraversableHandle handle;
    CUdeviceptr d_output_buffer;
    uint32_t num_sbt_records;
};

// Instance acceleration structure (IAS) 用
// 
struct InstanceAccelData
{
    OptixTraversableHandle handle;
    CUdeviceptr d_output_buffer;
    
    // so that we can update the data in the OptixInstance building the IAS,
    // store the pointer on the device side
    CUdeviceptr d_instances_buffer;
};

enum class ShapeType
{
    Mesh, 
    Sphere
};

struct OneWeekendState
{
    OptixDeviceContext context = 0;

    // Instance acceleration structure for the whole scene
    InstanceAccelData           ias                      = {};
    // A pointer to an array containing all the scene sphere data on the GPU
    void*                       d_sphere_data            = nullptr;
    // A pointer to an array containing all scene triangle data on the GPU
    void*                       d_mesh_data              = nullptr;

    OptixModule                 module                   = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixPipeline               pipeline                 = nullptr;

    // Ray generation program
    OptixProgramGroup           raygen_prg               = nullptr;
    // Miss program
    OptixProgramGroup           miss_prg                 = nullptr;

    // HitGroup program for spheres
    OptixProgramGroup           sphere_hitgroup_prg      = nullptr;
    // HitGroup program for mesh
    OptixProgramGroup           mesh_hitgroup_prg        = nullptr;

    // Callable program for materials
    // OptiX supports derived class function calls (polymorphism) through base class pointers.
    // Forbidden, use Callable functions to achieve pseudo-polymorphism
    // Here, three types of Lambertian, Dielectric, and Metal are implemented.
    CallableProgram             lambertian_prg           = {};
    CallableProgram             dielectric_prg           = {};
    CallableProgram             metal_prg                = {};

    // Callable program for textures
    // Constant ... Solid Color, Checker ... Checkerboard
    CallableProgram             constant_prg             = {};
    CallableProgram             checker_prg              = {};

    // CUDA stream
    CUstream                    stream                   = 0;

    // Pipeline launch parameters
    // CUDA extern "C" __constant__ Params params
    // can be accessed from all modules by declaration
    Params                      params;
    Params*                     d_params;

    // Shader binding table
    OptixShaderBindingTable     sbt                      = {};
};

// GLFW callbacks ------------------------------------------------
static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    double xpos, ypos; 

    glfwGetCursorPos(window, &xpos, &ypos);

    if (action == GLFW_PRESS)
    {
        mouse_button = button; 
        trackball.startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
    }
    else 
    {
        mouse_button = -1;
    }
}

// -----------------------------------------------------------------------
static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    Params* params = static_cast<Params*>(glfwGetWindowUserPointer(window));

    // If the mouse moves while left-clicking, fix the gaze point and move the camera
    if (mouse_button == GLFW_MOUSE_BUTTON_LEFT)
    {
        trackball.setViewMode(sutil::Trackball::LookAtFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
        camera_changed = true;
    }
    // If the mouse moves while right-clicking, fix the camera origin and move the gaze point
    else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        trackball.setViewMode(sutil::Trackball::EyeFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
        camera_changed = true;
    }
}

// -----------------------------------------------------------------------
static void windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y )
{
    // Save window resolution before minimization when window is minimized
    if( minimized )
        return;

    // Ensure window size is at least 1 x 1
    sutil::ensureMinimumSize( res_x, res_y );

    Params* params = static_cast<Params*>( glfwGetWindowUserPointer( window ) );
    params->width  = res_x;
    params->height = res_y;
    camera_changed = true;
    resize_dirty   = true;
}

// -----------------------------------------------------------------------
static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    minimized = ( iconified > 0 );
}

// -----------------------------------------------------------------------
static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    if( action == GLFW_PRESS )
    {
        // Q or Esc -> 終了
        if( key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE )
        {
            glfwSetWindowShouldClose( window, true );
        }
    }
    else if( key == GLFW_KEY_G )
    {
        // toggle UI draw
    }
}

// -----------------------------------------------------------------------
static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    if( trackball.wheelEvent( (int)yscroll ) )
        camera_changed = true;
}

// -----------------------------------------------------------------------
OptixAabb sphereBound(const SphereData& sphere)
{
    // 球体のAxis-aligned bounding box (AABB)を返す
    const float3 center = sphere.center;
    const float radius = sphere.radius;
    return OptixAabb {
        /* minX = */ center.x - radius, /* minY = */ center.y - radius, /* minZ = */ center.z - radius, 
        /* maxX = */ center.x + radius, /* maxY = */ center.y + radius, /* maxZ = */ center.z + radius
    };
}

// -----------------------------------------------------------------------
void printUsageAndExit(const char* argv0)
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "         --launch-samples | -s       Number of samples per pixel per launch (default 16)\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 768x768\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit(0);
}

// -----------------------------------------------------------------------
// Initialize pipeline launch parameter
// -----------------------------------------------------------------------
void initLaunchParams( OneWeekendState& state )
{
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state.params.accum_buffer), 
        state.params.width * state.params.height * sizeof(float4)
    ));
    state.params.frame_buffer = nullptr;

    state.params.samples_per_launch = samples_per_launch;
    state.params.subframe_index = 0u;
    state.params.max_depth = 5;

    // Set traversableHandle for raytracing AS
    state.params.handle = state.ias.handle;

    CUDA_CHECK(cudaStreamCreate(&state.stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_params), sizeof(Params)));
}

// -----------------------------------------------------------------------
// Camera update process
// Update the launch parameter when the camera moves due to mouse input etc.
// -----------------------------------------------------------------------
void handleCameraUpdate( Params& params )
{
    if (!camera_changed)
        return;

    camera_changed = false;
    camera.setAspectRatio(static_cast<float>(params.width) / static_cast<float>(params.height));
    params.eye = camera.eye();
    camera.UVWFrame(params.U, params.V, params.W);
}

// -----------------------------------------------------------------------
// Handling when the window size changes
// Update the buffer that stores the results of ray tracing calculations
// -----------------------------------------------------------------------
void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params )
{
    if( !resize_dirty )
        return;
    resize_dirty = false;

    output_buffer.resize( params.width, params.height );

    // Realloc accumulation buffer
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params.accum_buffer ) ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &params.accum_buffer ),
                params.width * params.height * sizeof( float4 )
                ) );
}

// -----------------------------------------------------------------------
// Monitor camera and window size changes
// -----------------------------------------------------------------------
void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params )
{
    // Update params on device
    if( camera_changed || resize_dirty )
        params.subframe_index = 0;

    handleCameraUpdate( params );
    handleResize( output_buffer, params );
}

// -----------------------------------------------------------------------
// Call optixLaunch to launch the device-side ray tracing kernel
// -----------------------------------------------------------------------
void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, OneWeekendState& state )
{
    uchar4* result_buffer_data = output_buffer.map();
    state.params.frame_buffer  = result_buffer_data;
    CUDA_CHECK( cudaMemcpyAsync(
                reinterpret_cast<void*>( state.d_params ),
                &state.params, sizeof( Params ),
                cudaMemcpyHostToDevice, state.stream
                ) );

    OPTIX_CHECK( optixLaunch(
                state.pipeline,
                state.stream,
                reinterpret_cast<CUdeviceptr>( state.d_params ),
                sizeof( Params ),
                &state.sbt,
                state.params.width,   // launch width
                state.params.height,  // launch height
                1                     // launch depth
                ) );
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}

// -----------------------------------------------------------------------
// Draw rendered results via OpenGL
// -----------------------------------------------------------------------
void displaySubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window )
{
    // Display
    int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;  //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display(
            output_buffer.width(),
            output_buffer.height(),
            framebuf_res_x,
            framebuf_res_y,
            output_buffer.getPBO()
            );
}

// -----------------------------------------------------------------------
// Callable function for getting messages from the device side
// When creating OptixDeviceContext, 
// Register with OptixDeviceContext::logCallbackFunction
// -----------------------------------------------------------------------
static void contextLogCallback(uint32_t level, const char* tag, const char* msg, void* /* callback_data */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << msg << "\n";
}

// -----------------------------------------------------------------------
// camera initialization
// -----------------------------------------------------------------------
void initCameraState()
{
    camera_changed = true;

    camera.setEye(make_float3(13.0f, 2.0f, 3.0f));
    camera.setLookat(make_float3(0.0f, 0.0f, 0.0f));
    camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
    camera.setFovY(20.0f);
    camera.setAspectRatio(3.0f / 2.0f);

    trackball.setCamera(&camera);
    trackball.setMoveSpeed(10.0f);
    trackball.setReferenceFrame(
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 1.0f),
        make_float3(0.0f, 1.0f, 0.0f)
    );
    trackball.setGimbalLock(true);
}

// -----------------------------------------------------------------------
// Initialize OptixDeviceContext
// -----------------------------------------------------------------------
void createContext( OneWeekendState& state )
{
    // CUDA initialization
    CUDA_CHECK( cudaFree( 0 ) );

    OptixDeviceContext context;
    CUcontext   cu_ctx = 0;
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction  = &contextLogCallback;
    // Level of message to get in Callback
    // 0 ... disable, don't receive messages
    // 1 ... fatal, Unrecoverable error. Context or OptiX disabled
    // 2 ... error, Correctable error.
    // 3 ... warning, It warns you about unintended behavior or leads to poor performance.
    // 4 ... print, receive all messages
    options.logCallbackLevel     = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &options, &context ) );

    state.context = context;
}

// -----------------------------------------------------------------------
// Count the number of unique indexes
// (for example) { 0, 0, 0, 1, 1, 2, 2, 2 } -> 3 
// -----------------------------------------------------------------------
uint32_t getNumSbtRecords(const std::vector<uint32_t>& sbt_indices)
{
    std::vector<uint32_t> sbt_counter;
    for (const uint32_t& sbt_idx : sbt_indices)
    {
        auto itr = std::find(sbt_counter.begin(), sbt_counter.end(), sbt_idx);
        if (sbt_counter.empty() || itr == sbt_counter.end())
            sbt_counter.emplace_back(sbt_idx);
    }
    return static_cast<uint32_t>(sbt_counter.size());
}

// -----------------------------------------------------------------------
// Building a Geometry acceleration structure
// -----------------------------------------------------------------------
void buildGAS( OneWeekendState& state, GeometryAccelData& gas, OptixBuildInput& build_input)
{
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION; // Allow compaction after build
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;        // OPERATION_UPDATE when updating AS

    // Calculate the memory area required for AS build
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
        state.context, 
        &accel_options, 
        &build_input, 
        1, 
        &gas_buffer_sizes
    ));

    // Allocate a temporary buffer for building AS
    CUdeviceptr d_temp_buffer;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer ), gas_buffer_sizes.tempSizeInBytes ) );

    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compacted_size_offset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size), 
        compacted_size_offset + 8
    ));

    // Emit property to secure the data area after compaction
    OptixAccelEmitDesc emit_property = {};
    emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_property.result = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compacted_size_offset );

    // AS build
    OPTIX_CHECK(optixAccelBuild(
        state.context,
        state.stream,
        &accel_options,
        &build_input, 
        1, 
        d_temp_buffer, 
        gas_buffer_sizes.tempSizeInBytes, 
        d_buffer_temp_output_gas_and_compacted_size, 
        gas_buffer_sizes.outputSizeInBytes, 
        &gas.handle, 
        &emit_property, 
        1
    ));

    // free the temporary buffer as it is not needed
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emit_property.result, sizeof(size_t), cudaMemcpyDeviceToHost));
    // Perform compaction only if the area after compaction is smaller than the area size before compaction
    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gas.d_output_buffer), compacted_gas_size));
        OPTIX_CHECK(optixAccelCompact(state.context, 0, gas.handle, gas.d_output_buffer, compacted_gas_size, &gas.handle));
        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    }
    else
    {
        gas.d_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

// -----------------------------------------------------------------------
// build GAS for mesh
// Also copy data to device side pointer (state.d_mesh_data) at the same time
// -----------------------------------------------------------------------
void buildMeshGAS(
    OneWeekendState& state, 
    GeometryAccelData& gas,
    const std::vector<float3>& vertices, 
    const std::vector<uint3>& indices, 
    const std::vector<uint32_t>& sbt_indices
)
{
    // Copy the vertex information that makes up the mesh to the GPU
    CUdeviceptr d_vertices = 0;
    const size_t vertices_size = vertices.size() * sizeof(float3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_vertices), 
        vertices.data(), vertices_size, 
        cudaMemcpyHostToDevice
    ));

    // Copy index information that defines how to connect vertices to GPU
    CUdeviceptr d_indices = 0;
    const size_t indices_size = indices.size() * sizeof(uint3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices), indices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_indices),
        indices.data(), indices_size, 
        cudaMemcpyHostToDevice 
    ));
    
    // Store mesh data in structure and copy to GPU
    MeshData mesh_data{reinterpret_cast<float3*>(d_vertices), reinterpret_cast<uint3*>(d_indices) };
    CUDA_CHECK(cudaMalloc(&state.d_mesh_data, sizeof(MeshData)));
    CUDA_CHECK(cudaMemcpy(
        state.d_mesh_data, &mesh_data, sizeof(MeshData), cudaMemcpyHostToDevice
    ));

    // Copy array of sbt indices relative to Instance sbt offset to GPU
    CUdeviceptr d_sbt_indices = 0;
    const size_t sbt_indices_size = sbt_indices.size() * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbt_indices), sbt_indices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void**>(d_sbt_indices),
        sbt_indices.data(), sbt_indices_size,
        cudaMemcpyHostToDevice
    ));

    // Count the number of unique sbt_indexes
    uint32_t num_sbt_records = getNumSbtRecords(sbt_indices);
    gas.num_sbt_records = num_sbt_records;

    // Set flags for sbt_index with no duplication
    // Set to FLAG_NONE or FLAG_REQUIRE_SINGLE_ANYHIT_CALL if you want to use the Anyhit program
    uint32_t* input_flags = new uint32_t[num_sbt_records];
    for (uint32_t i = 0; i < num_sbt_records; i++)
        input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    // Set mesh vertex information, index buffer, SBT record index array to build input
    // Note that num_sbt_records is the number of SBT records, not the number of triangles.
    OptixBuildInput mesh_input = {};
    mesh_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    mesh_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    mesh_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    mesh_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
    mesh_input.triangleArray.vertexBuffers = &d_vertices;
    mesh_input.triangleArray.flags = input_flags;
    mesh_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    mesh_input.triangleArray.indexStrideInBytes = sizeof(uint3);
    mesh_input.triangleArray.indexBuffer = d_indices;
    mesh_input.triangleArray.numIndexTriplets = static_cast<uint32_t>(indices.size());
    mesh_input.triangleArray.numSbtRecords = num_sbt_records;
    mesh_input.triangleArray.sbtIndexOffsetBuffer = d_sbt_indices;
    mesh_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    mesh_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

    buildGAS(state, gas, mesh_input);
}

// -----------------------------------------------------------------------
// Building GAS for Sphere
// Also copy data to the device side pointer (state.d_sphere_data) at the same time
// -----------------------------------------------------------------------
void buildSphereGAS(
    OneWeekendState& state, 
    GeometryAccelData& gas,
    const std::vector<SphereData>& spheres, 
    const std::vector<uint32_t>& sbt_indices
)
{
    // Create AABB array from Sphere array
    std::vector<OptixAabb> aabb;
    std::transform(spheres.begin(), spheres.end(), std::back_inserter(aabb),
        [](const SphereData& sphere) { return sphereBound(sphere); });

    // Copy array of AABB to GPU
    CUdeviceptr d_aabb_buffer;
    const size_t aabb_size = sizeof(OptixAabb) * aabb.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb_buffer), aabb_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_aabb_buffer),
        aabb.data(), aabb_size,
        cudaMemcpyHostToDevice
    ));

    // Copy array of sbt indices relative to Instance sbt offset to GPU
    CUdeviceptr d_sbt_indices;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbt_indices), sizeof(uint32_t) * sbt_indices.size()));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_sbt_indices),
        sbt_indices.data(), sizeof(uint32_t) * sbt_indices.size(),
        cudaMemcpyHostToDevice
    ));

    // Copy array of global sphere data onto GPU
    // Access to individual sphere data is via optixGetPrimitiveIndex()
    CUDA_CHECK(cudaMalloc(&state.d_sphere_data, sizeof(SphereData) * spheres.size()));
    CUDA_CHECK(cudaMemcpy(state.d_sphere_data, spheres.data(), sizeof(SphereData) * spheres.size(), cudaMemcpyHostToDevice));

    // Count the number of unique sbt_indexes
    uint32_t num_sbt_records = getNumSbtRecords(sbt_indices);
    gas.num_sbt_records = num_sbt_records;

    // Set flags for sbt_index with no duplication
    // Set to FLAG_NONE or FLAG_REQUIRE_SINGLE_ANYHIT_CALL if you want to use the Anyhit program
    uint32_t* input_flags = new uint32_t[num_sbt_records];
    for (uint32_t i = 0; i < num_sbt_records; i++)
        input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    // AABB array for custom primitives and index array of SBT records
    // set in build input
    // Note that num_sbt_records is the number of SBT records, not the primitive number.
    OptixBuildInput sphere_input = {};
    sphere_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    sphere_input.customPrimitiveArray.aabbBuffers = &d_aabb_buffer;
    sphere_input.customPrimitiveArray.numPrimitives = static_cast<uint32_t>(spheres.size());
    sphere_input.customPrimitiveArray.flags = input_flags;
    sphere_input.customPrimitiveArray.numSbtRecords = num_sbt_records;
    sphere_input.customPrimitiveArray.sbtIndexOffsetBuffer = d_sbt_indices;
    sphere_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    sphere_input.customPrimitiveArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

    buildGAS(state, gas, sphere_input);
}

// -----------------------------------------------------------------------
// Building an instance acceleration structure
// -----------------------------------------------------------------------
void buildIAS(OneWeekendState& state, InstanceAccelData& ias, const std::vector<OptixInstance>& instances)
{
    CUdeviceptr d_instances;
    const size_t instances_size = sizeof(OptixInstance) * instances.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances), instances_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_instances),
        instances.data(), instances_size,
        cudaMemcpyHostToDevice
    ));

    OptixBuildInput instance_input = {};
    instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances = d_instances;
    instance_input.instanceArray.numInstances = static_cast<uint32_t>(instances.size());

    OptixAccelBuildOptions accel_options = {};
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &instance_input,
        1, // num build input
        &ias_buffer_sizes
    ));

    size_t d_temp_buffer_size = ias_buffer_sizes.tempSizeInBytes;

    // Allocate a temporary buffer for building AS
    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_temp_buffer),
        d_temp_buffer_size
    ));

    CUdeviceptr d_buffer_temp_output_ias_and_compacted_size;
    size_t compacted_size_offset = roundUp<size_t>(ias_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_buffer_temp_output_ias_and_compacted_size),
        compacted_size_offset + 8
    ));

    // Emit property to secure the data area after compaction
    OptixAccelEmitDesc emit_property = {};
    emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_property.result = ( CUdeviceptr )( (char*)d_buffer_temp_output_ias_and_compacted_size + compacted_size_offset );

    // AS build
    OPTIX_CHECK(optixAccelBuild(
        state.context,
        state.stream,
        &accel_options,
        &instance_input,
        1,                  // num build inputs
        d_temp_buffer,
        d_temp_buffer_size,
        // ias.d_output_buffer,
        d_buffer_temp_output_ias_and_compacted_size,
        ias_buffer_sizes.outputSizeInBytes,
        &ias.handle,        // emitted property list
        nullptr,            // num emitted property
        0
    ));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));

    size_t compacted_ias_size;
    CUDA_CHECK(cudaMemcpy(&compacted_ias_size, (void*)emit_property.result, sizeof(size_t), cudaMemcpyDeviceToHost));
    if (compacted_ias_size < ias_buffer_sizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ias.d_output_buffer), compacted_ias_size));
        OPTIX_CHECK(optixAccelCompact(state.context, 0, ias.handle, ias.d_output_buffer, compacted_ias_size, &ias.handle));
        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_ias_and_compacted_size));
    }
    else
    {
        ias.d_output_buffer = d_buffer_temp_output_ias_and_compacted_size;
    }
}

// -----------------------------------------------------------------------
// Creating OptixModules
// -----------------------------------------------------------------------
void createModule(OneWeekendState& state)
{
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    // ~7.3 series OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    state.pipeline_compile_options.numPayloadValues = 2;
    // Attribute number setting
    // Since the intersection detection of Sphere passes the normal and texture coordinates to intersection -> closesthit
    // (x, y, z) ... 3D, (s, t) ... 2D with total of 5 Attributes required
    // optixinOneWeekend.cu: see line 339
    state.pipeline_compile_options.numAttributeValues = 5;
#ifdef DEBUG 
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    // Variable name for Pipeline launch parameter
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    size_t      inputSize = 0;
    const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixInOneWeekend.cu", inputSize);

    // Create Module from PTX
    char   log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        state.context,
        &module_compile_options,
        &state.pipeline_compile_options,
        input,
        inputSize,
        log,
        &sizeof_log,
        &state.module
    ));
}

// -----------------------------------------------------------------------
// Generate a direct callable program. Increment callable_id by 1 for each generation
// -----------------------------------------------------------------------
void createDirectCallables(const OneWeekendState& state, CallableProgram& callable, const char* dc_function_name, uint32_t& callables_id)
{
    OptixProgramGroupOptions prg_options = {};

    OptixProgramGroupDesc callables_prg_desc = {};

    char log[2048];
    size_t sizeof_log = sizeof(log);

    callables_prg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    callables_prg_desc.callables.moduleDC = state.module;
    callables_prg_desc.callables.entryFunctionNameDC = dc_function_name;
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &callables_prg_desc,
        1,
        &prg_options,
        log,
        &sizeof_log,
        &callable.program
    ));
    callable.id = callables_id;
    callables_id++;
}

// -----------------------------------------------------------------------
// Create all ProgramGroups
// -----------------------------------------------------------------------
void createProgramGroups(OneWeekendState& state)
{
    OptixProgramGroupOptions prg_options = {};

    char log[2048];
    size_t sizeof_log = sizeof(log);

    // Raygen program
    {
        OptixProgramGroupDesc raygen_prg_desc = {};
        raygen_prg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prg_desc.raygen.module = state.module;
        raygen_prg_desc.raygen.entryFunctionName = "__raygen__pinhole";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context, 
            &raygen_prg_desc, 
            1, // num program groups
            &prg_options, 
            log, 
            &sizeof_log, 
            &state.raygen_prg
        ));
    }

    // Miss program
    {
        OptixProgramGroupDesc miss_prg_desc = {};
        miss_prg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prg_desc.miss.module = state.module;
        miss_prg_desc.miss.entryFunctionName = "__miss__radiance";
        sizeof_log = sizeof(log);

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context, 
            &miss_prg_desc, 
            1, 
            &prg_options, 
            log, 
            &sizeof_log, 
            &state.miss_prg
        ));
    }

    // Hitgroup programs
    {
        // Mesh
        OptixProgramGroupDesc hitgroup_prg_desc = {};
        hitgroup_prg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prg_desc.hitgroup.moduleCH = state.module;
        hitgroup_prg_desc.hitgroup.entryFunctionNameCH = "__closesthit__mesh";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            &hitgroup_prg_desc,
            1,
            &prg_options,
            log,
            &sizeof_log,
            &state.mesh_hitgroup_prg
        ));

        // Sphere
        memset(&hitgroup_prg_desc, 0, sizeof(OptixProgramGroupDesc));
        hitgroup_prg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prg_desc.hitgroup.moduleIS = state.module;
        hitgroup_prg_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
        hitgroup_prg_desc.hitgroup.moduleCH = state.module;
        hitgroup_prg_desc.hitgroup.entryFunctionNameCH = "__closesthit__sphere";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            &hitgroup_prg_desc,
            1,
            &prg_options,
            log,
            &sizeof_log,
            &state.sphere_hitgroup_prg
        ));
    }

    uint32_t callables_id = 0;
    // Callable program for material
    {
        // Lambertian
        createDirectCallables(state, state.lambertian_prg, "__direct_callable__lambertian", callables_id);
        // Dielectric
        createDirectCallables(state, state.dielectric_prg, "__direct_callable__dielectric", callables_id);
        // Metal
        createDirectCallables(state, state.metal_prg, "__direct_callable__metal", callables_id);
    }

    // Callable program for textures
    {
        // Constant texture
        createDirectCallables(state, state.constant_prg, "__direct_callable__constant", callables_id);
        // Checker texture
        createDirectCallables(state, state.checker_prg, "__direct_callable__checker", callables_id);
    }
}

// -----------------------------------------------------------------------
// Creating an OptixPipeline
// -----------------------------------------------------------------------
void createPipeline(OneWeekendState& state)
{
    OptixProgramGroup program_groups[] =
    {
        state.raygen_prg, 
        state.miss_prg, 
        state.mesh_hitgroup_prg, 
        state.sphere_hitgroup_prg, 
        state.lambertian_prg.program, 
        state.dielectric_prg.program,
        state.metal_prg.program, 
        state.constant_prg.program, 
        state.checker_prg.program
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    // set call depth for optixTrace()
    pipeline_link_options.maxTraceDepth = 2;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        log,
        &sizeof_log,
        &state.pipeline
    ));

    // Compute stack size of Call graph built by pipeline from each program
    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.raygen_prg, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.miss_prg, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.mesh_hitgroup_prg, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.sphere_hitgroup_prg, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.lambertian_prg.program, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.dielectric_prg.program, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.metal_prg.program, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.constant_prg.program, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.checker_prg.program, &stack_sizes));

    uint32_t max_trace_depth = pipeline_link_options.maxTraceDepth;
    // Continuation callable is not used, so 0 is fine
    uint32_t max_cc_depth = 0;
    // Direct callable call depth is at most 2 (Material -> Texture)
    uint32_t max_dc_depth = 3;
    uint32_t direct_callable_stack_size_from_traversable;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes, 
        max_trace_depth, 
        max_cc_depth, 
        max_dc_depth,
        &direct_callable_stack_size_from_traversable,
        &direct_callable_stack_size_from_state, 
        &continuation_stack_size
    ));

    // set the depth of the traversable graph
    // If it ends only with IAS -> GAS like this time, the depth of the traversable graph will be 2
    // If IAS -> Motion transform -> GAS, depth should be 3
    const uint32_t max_traversal_depth = 2;
    OPTIX_CHECK(optixPipelineSetStackSize(
        state.pipeline, 
        direct_callable_stack_size_from_traversable, 
        direct_callable_stack_size_from_state,
        continuation_stack_size, 
        max_traversal_depth
    ));
}

// -----------------------------------------------------------------------
// Construction of shader binding table
// -----------------------------------------------------------------------
void createSBT(OneWeekendState& state, const std::vector<std::pair<ShapeType, HitGroupData>>& hitgroup_datas)
{
    // Ray generation 
    RayGenRecord raygen_record = {};
    // Allocate RayGenRecord area on the device side
    CUdeviceptr d_raygen_record;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), sizeof(RayGenRecord)));
    // Fill SBT record headers programmatically
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prg, &raygen_record));
    // Copy RayGenRecord to device side
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_raygen_record),
        &raygen_record,
        sizeof(RayGenRecord),
        cudaMemcpyHostToDevice
    ));

    // Miss
    MissRecord miss_record = {};
    // Allocate space for MissRecord on the device side
    CUdeviceptr d_miss_record;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), sizeof(MissRecord)));
    // Fill SBT record headers programmatically
    OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_prg, &miss_record));
    // set data
    miss_record.data.bg_color = make_float4(0.0f);
    // Copy MissRecord to device side
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_miss_record),
        &miss_record,
        sizeof(MissRecord),
        cudaMemcpyHostToDevice
    ));

    // HitGroup
    HitGroupRecord* hitgroup_records = new HitGroupRecord[hitgroup_datas.size()];
    // Allocate an area for HitGroupRecord on the device side
    CUdeviceptr d_hitgroup_records;
    const size_t hitgroup_record_size = sizeof(HitGroupRecord) * hitgroup_datas.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_records), hitgroup_record_size));

    // Build Shader binding table from HitGroupData
    for (size_t i = 0; i < hitgroup_datas.size(); i++)
    {
        ShapeType type = hitgroup_datas[i].first;
        HitGroupData data = hitgroup_datas[i].second;
        // switch the program for filling the header depending on the ShapeType
        if (type == ShapeType::Mesh)
            OPTIX_CHECK(optixSbtRecordPackHeader(state.mesh_hitgroup_prg, &hitgroup_records[i]));
        else if (type == ShapeType::Sphere)
            OPTIX_CHECK(optixSbtRecordPackHeader(state.sphere_hitgroup_prg, &hitgroup_records[i]));
        // set data
        hitgroup_records[i].data = data;
    }
    // Copy HitGroupRecord to device
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_hitgroup_records),
        hitgroup_records,
        hitgroup_record_size,
        cudaMemcpyHostToDevice
    ));

    // Since it is not necessary to register data in the Shader binding table of the Callable program this time,
    // Copy empty data using EmptyRecord.
    // However, the header must be filled programmatically even if there is no data.
    // If you forget this, Invalid memory access will occur after starting raytracing.
    // It is difficult to notice in debugging, so be careful
    EmptyRecord* callables_records = new EmptyRecord[5];
    CUdeviceptr d_callables_records;
    const size_t callables_record_size = sizeof(EmptyRecord) * 5;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_callables_records), callables_record_size));

    OPTIX_CHECK(optixSbtRecordPackHeader(state.lambertian_prg.program, &callables_records[state.lambertian_prg.id]));
    OPTIX_CHECK(optixSbtRecordPackHeader(state.dielectric_prg.program, &callables_records[state.dielectric_prg.id]));
    OPTIX_CHECK(optixSbtRecordPackHeader(state.metal_prg.program, &callables_records[state.metal_prg.id]));
    OPTIX_CHECK(optixSbtRecordPackHeader(state.constant_prg.program, &callables_records[state.constant_prg.id]));
    OPTIX_CHECK(optixSbtRecordPackHeader(state.checker_prg.program, &callables_records[state.checker_prg.id]));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_callables_records),
        callables_records,
        callables_record_size,
        cudaMemcpyHostToDevice
    ));

    // Build Shader binding table from each record
    // Here, set the pointer to the beginning of the record array, the alignment of the shader binding table, and the number of arrays
    state.sbt.raygenRecord = d_raygen_record;
    state.sbt.missRecordBase = d_miss_record;
    state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(sizeof(MissRecord));
    state.sbt.missRecordCount = 1;
    state.sbt.hitgroupRecordBase = d_hitgroup_records;
    state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof(HitGroupRecord));
    state.sbt.hitgroupRecordCount = static_cast<uint32_t>(hitgroup_datas.size());
    state.sbt.callablesRecordBase = d_callables_records;
    state.sbt.callablesRecordCount = 5;
    state.sbt.callablesRecordStrideInBytes = sizeof(EmptyRecord);
}

// -----------------------------------------------------------------------
void finalizeState(OneWeekendState& state)
{
    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prg));
    OPTIX_CHECK(optixProgramGroupDestroy(state.miss_prg));
    OPTIX_CHECK(optixProgramGroupDestroy(state.mesh_hitgroup_prg));
    OPTIX_CHECK(optixProgramGroupDestroy(state.sphere_hitgroup_prg));
    OPTIX_CHECK(optixProgramGroupDestroy(state.lambertian_prg.program));
    OPTIX_CHECK(optixProgramGroupDestroy(state.dielectric_prg.program));
    OPTIX_CHECK(optixProgramGroupDestroy(state.metal_prg.program));
    OPTIX_CHECK(optixProgramGroupDestroy(state.constant_prg.program));
    OPTIX_CHECK(optixProgramGroupDestroy(state.checker_prg.program));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.hitgroupRecordBase)));
}

// -----------------------------------------------------------------------
// copy the data on the device and return the pointer as a generic pointer
// -----------------------------------------------------------------------
template <typename T>
void* copyDataToDevice(T data, size_t size)
{
    CUdeviceptr device_ptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_ptr), size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(device_ptr),
        &data, size,
        cudaMemcpyHostToDevice
    ));
    return reinterpret_cast<void*>(device_ptr);
}

// -----------------------------------------------------------------------
void createScene(OneWeekendState& state)
{
    // An array containing HitGroupData and material data
    // In this case, the sphere and mesh each use the same array for geometry data,
    // Switch intersecting data with optixGetPrimitiveIndex() on the device side
    // The material data adopts a method in which different data are distributed.
    // So, match the number of hitgroup_datas to the number of materials
    // <- It is enough to have HitGroupRecords for the number of materials, so there is no need to prepare the number of geometries
    std::vector<std::pair<ShapeType, HitGroupData>> hitgroup_datas;
    std::vector<Material> materials;

    // --------------------------------------------------------------------
    // sphere scene building
    // Assume that the spheres all have different materials
    // --------------------------------------------------------------------
    // Data preparation for sphere
    std::vector<SphereData> spheres;
    // Array of relative sbt_indexes for spheres
    std::vector<uint32_t> sphere_sbt_indices;
    uint32_t sphere_sbt_index = 0;

    // Ground
    SphereData ground_sphere{ make_float3(0, -1000, 0), 1000 };
    spheres.emplace_back(ground_sphere);
    // texture
    CheckerData ground_checker{ make_float4(1.0f), make_float4(0.2f, 0.5f, 0.2f, 1.0f), 5000};
    // Lambertian material
    LambertianData ground_lambert{ copyDataToDevice(ground_checker, sizeof(CheckerData)), state.checker_prg.id };
    materials.push_back(Material{ copyDataToDevice(ground_lambert, sizeof(LambertianData)), state.lambertian_prg.id });
    // Added sbt_index because I added material
    sphere_sbt_indices.emplace_back(sphere_sbt_index++);
    
    // Generate Seed Value for Pseudorandom Numbers
    uint32_t seed = tea<4>(0, 0);
    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            const float choose_mat = rnd(seed);
            const float3 center{ a + 0.9f * rnd(seed), 0.2f, b + 0.9f * rnd(seed) };
            if (length(center - make_float3(4, 0.2, 0)) > 0.9f)
            {
                // add sphere
                spheres.emplace_back( SphereData { center, 0.2f });

                // Probabilistically create Lambertian, Metal, and Dielectric materials
                // Allocate a Callable program ID according to the type when adding
                if (choose_mat < 0.8f)
                {
                    // Lambertian
                    ConstantData albedo{ make_float4(rnd(seed), rnd(seed), rnd(seed), 1.0f) };
                    LambertianData lambertian{ copyDataToDevice(albedo, sizeof(ConstantData)), state.constant_prg.id };
                    materials.emplace_back(Material{ copyDataToDevice(lambertian, sizeof(LambertianData)), state.lambertian_prg.id });
                }
                else if (choose_mat < 0.95f)
                {
                    // Metal
                    ConstantData albedo{ make_float4(0.5f + rnd(seed) * 0.5f) };
                    MetalData metal{ copyDataToDevice(albedo, sizeof(ConstantData)), state.constant_prg.id, /* fuzz = */ rnd(seed) * 0.5f};
                    materials.emplace_back(Material{ copyDataToDevice(metal, sizeof(MetalData)), state.metal_prg.id });
                }
                else
                {
                    // Dielectric
                    ConstantData albedo{ make_float4(1.0f) };
                    DielectricData glass{ copyDataToDevice(albedo, sizeof(ConstantData)), state.constant_prg.id, /* ior = */ 1.5f};
                    materials.emplace_back(Material{ copyDataToDevice(glass, sizeof(DielectricData)), state.dielectric_prg.id });
                }
                sphere_sbt_indices.emplace_back(sphere_sbt_index++);
            }
        }
    }
    
    // Dielectric
    spheres.emplace_back(SphereData{ make_float3(0.0f, 1.0f, 0.0f), 1.0f });
    ConstantData albedo1{ make_float4(1.0f) };
    DielectricData material1{ copyDataToDevice(albedo1, sizeof(ConstantData)), state.constant_prg.id, /* ior = */ 1.5f };
    materials.push_back(Material{ copyDataToDevice(material1, sizeof(DielectricData)), state.dielectric_prg.id });
    sphere_sbt_indices.emplace_back(sphere_sbt_index++);

    // Lambertian
    spheres.emplace_back(SphereData{ make_float3(-4.0f, 1.0f, 0.0f), 1.0f });
    ConstantData albedo2{ make_float4(0.4f, 0.2f, 0.1f, 1.0f) };
    LambertianData material2{ copyDataToDevice(albedo2, sizeof(ConstantData)), state.constant_prg.id };
    materials.push_back(Material{ copyDataToDevice(material2, sizeof(LambertianData)), state.lambertian_prg.id });
    sphere_sbt_indices.emplace_back(sphere_sbt_index++);

    // Metal
    spheres.emplace_back(SphereData{ make_float3(4.0f, 1.0f, 0.0f), 1.0f });
    ConstantData albedo3{ make_float4(0.7f, 0.6f, 0.5f, 1.0f) };
    MetalData material3{ copyDataToDevice(albedo3, sizeof(ConstantData)), state.constant_prg.id };
    materials.emplace_back(Material{ copyDataToDevice(material3, sizeof(MetalData)), state.metal_prg.id });
    sphere_sbt_indices.emplace_back(sphere_sbt_index++);

    // Create GAS for Sphere (Data is also copied to state.d_sphere_data internally at the same time)
    GeometryAccelData sphere_gas;
    buildSphereGAS(state, sphere_gas, spheres, sphere_sbt_indices);

    // Prepare data for Shader binding table from array of material and sphere data
    for (auto& m : materials)
        hitgroup_datas.emplace_back(ShapeType::Sphere, HitGroupData{state.d_sphere_data, m});

    // --------------------------------------------------------------------
    // Mesh scene building
    // Only 3 types of materials are allocated for 100 triangles in mesh
    // Mesh data is common to all materials, so only 3 SBT records are required.
    // --------------------------------------------------------------------
    std::vector<float3> mesh_vertices;
    std::vector<uint3> mesh_indices;
    std::vector<uint32_t> mesh_sbt_indices;
    uint32_t mesh_index = 0;
    for (int a = 0; a < 100; a++) {
        float3 center{rnd(seed) * 20.0f - 10.0f, 0.5f + rnd(seed) * 1.0f - 0.5f, rnd(seed) * 20.0f - 10.0f };
        const float3 p0 = center + make_float3(rnd(seed) * 0.5f, -rnd(seed) * 0.5f, rnd(seed) * 0.5f - 0.25f);
        const float3 p1 = center + make_float3(-rnd(seed) * 0.5f, -rnd(seed) * 0.5f, rnd(seed) * 0.5f - 0.25f);
        const float3 p2 = center + make_float3(rnd(seed) * 0.25f, rnd(seed) * 0.5f, rnd(seed) * 0.5f - 0.25f);

        mesh_vertices.emplace_back(p0);
        mesh_vertices.emplace_back(p1);
        mesh_vertices.emplace_back(p2);
        mesh_indices.emplace_back(make_uint3(mesh_index + 0, mesh_index + 1, mesh_index + 2));
        mesh_index += 3;
    }

    const uint32_t red_sbt_index = 0;
    const uint32_t green_sbt_index = 1;
    const uint32_t blue_sbt_index = 2;

    // Randomly assign three colors of red, green, and blue
    for (size_t i = 0; i < mesh_indices.size(); i++)
    {
        const float choose_rgb = rnd(seed);
        if (choose_rgb < 0.33f)
            mesh_sbt_indices.push_back(red_sbt_index);
        else if (choose_rgb < 0.67f)
            mesh_sbt_indices.push_back(green_sbt_index);
        else
            mesh_sbt_indices.push_back(blue_sbt_index);
    }

    // Create gas for mesh
    GeometryAccelData mesh_gas;
    buildMeshGAS(state, mesh_gas, mesh_vertices, mesh_indices, mesh_sbt_indices);

    // Prepare red, green and blue materials and add HitGroupData
    // red
    ConstantData red{ {0.8f, 0.05f, 0.05f, 1.0f} };
    LambertianData red_lambert{ copyDataToDevice(red, sizeof(ConstantData)), state.constant_prg.id };
    materials.emplace_back(Material{ copyDataToDevice(red_lambert, sizeof(LambertianData)), state.lambertian_prg.id });
    hitgroup_datas.emplace_back(ShapeType::Mesh, HitGroupData{ state.d_mesh_data, materials.back() });

    // green
    ConstantData green{ {0.05f, 0.8f, 0.05f, 1.0f} };
    LambertianData green_lambert{ copyDataToDevice(green, sizeof(ConstantData)), state.constant_prg.id };
    materials.emplace_back(Material{ copyDataToDevice(green_lambert, sizeof(LambertianData)), state.lambertian_prg.id });
    hitgroup_datas.emplace_back(ShapeType::Mesh, HitGroupData{ state.d_mesh_data, materials.back() });

    // blue
    ConstantData blue{ {0.05f, 0.05f, 0.8f, 1.0f} };
    LambertianData blue_lambert{ copyDataToDevice(blue, sizeof(ConstantData)), state.constant_prg.id };
    materials.emplace_back(Material{ copyDataToDevice(blue_lambert, sizeof(LambertianData)), state.lambertian_prg.id });
    hitgroup_datas.emplace_back(ShapeType::Mesh, HitGroupData{ state.d_mesh_data, materials.back() });

    // Create Instance for IAS for sphere and mesh
    std::vector<OptixInstance> instances;
    uint32_t flags = OPTIX_INSTANCE_FLAG_NONE;

    uint32_t sbt_offset = 0;
    uint32_t instance_id = 0;
    instances.emplace_back(OptixInstance{
        {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, instance_id, sbt_offset, 255, 
        flags, sphere_gas.handle, {0, 0}
    });

    sbt_offset += sphere_gas.num_sbt_records;
    instance_id++;
    // Rotate the mesh by PI/6 around the Y axis
    const float c = cosf(M_PIf / 6.0f);
    const float s = sinf(M_PIf / 6.0f);
    instances.push_back(OptixInstance{
        {c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0}, instance_id, sbt_offset, 255,
        flags, mesh_gas.handle, {0, 0}
    });

    // create IAS
    buildIAS(state, state.ias, instances);

    // create shader binding table
    createSBT(state, hitgroup_datas);
}

// -----------------------------------------------------------------------
int main(int argc, char* argv[])
{
    OneWeekendState state;
    state.params.width = 1200;
    state.params.height = static_cast<int>(1200.0f / (3.0f / 2.0f));
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    std::string outfile;

    for (int i = 1; i < argc; i++)
    {
        const std::string arg = argv[i];
        if (arg == "--file" || arg == "-f")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            outfile = argv[++i];
        }
        else if (arg.substr(0, 6) == "--dim=")
        {
            const std::string dims_arg = arg.substr(6);
            int w, h;
            sutil::parseDimensions(dims_arg.c_str(), w, h);
            state.params.width = w;
            state.params.height = h;
        }
        else if (arg == "--launch-samples" || arg == "-s")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            samples_per_launch = atoi(argv[++i]);
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit(argv[0]);
        }
    }

    try
    {
        initCameraState();

        createContext(state);
        createModule(state);
        createProgramGroups(state);
        createPipeline(state);
        createScene(state);
        initLaunchParams(state);

        if (outfile.empty())
        {
            GLFWwindow* window = sutil::initUI("optixInOneWeekend", state.params.width, state.params.height);
            glfwSetMouseButtonCallback(window, mouseButtonCallback);
            glfwSetCursorPosCallback(window, cursorPosCallback);
            glfwSetWindowSizeCallback(window, windowSizeCallback);
            glfwSetWindowIconifyCallback(window, windowIconifyCallback);
            glfwSetKeyCallback(window, keyCallback);
            glfwSetScrollCallback(window, scrollCallback);
            glfwSetWindowUserPointer(window, &state.params);

            //
            // Render loop
            //
            {
                sutil::CUDAOutputBuffer<uchar4> output_buffer(
                    output_buffer_type,
                    state.params.width,
                    state.params.height
                );

                output_buffer.setStream(state.stream);
                sutil::GLDisplay gl_display;

                std::chrono::duration<double> state_update_time(0.0);
                std::chrono::duration<double> render_time(0.0);
                std::chrono::duration<double> display_time(0.0);

                do
                {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    updateState(output_buffer, state.params);
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    launchSubframe(output_buffer, state);
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;

                    displaySubframe(output_buffer, gl_display, window);
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    sutil::displayStats(state_update_time, render_time, display_time);

                    glfwSwapBuffers(window);

                    ++state.params.subframe_index;
                } while (!glfwWindowShouldClose(window));
                CUDA_SYNC_CHECK();
            }

            sutil::cleanupUI(window);
        }
        else
        {
            if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
            {
                sutil::initGLFW();  // For GL context
                sutil::initGL();
            }

            sutil::CUDAOutputBuffer<uchar4> output_buffer(
                output_buffer_type,
                state.params.width,
                state.params.height
            );

            handleCameraUpdate(state.params);
            handleResize(output_buffer, state.params);
            for (int i = 0; i < 1024; i += samples_per_launch) {
                launchSubframe(output_buffer, state);
                state.params.subframe_index++;
            }

            sutil::ImageBuffer buffer;
            buffer.data = output_buffer.getHostPointer();
            buffer.width = output_buffer.width();
            buffer.height = output_buffer.height();
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

            sutil::saveImage(outfile.c_str(), buffer, false);

            if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
            {
                glfwTerminate();
            }
        }

        finalizeState(state);
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}