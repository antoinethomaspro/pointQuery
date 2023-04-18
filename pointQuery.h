#pragma once

#include <vector>
#include "CUDABuffer.h"


struct Params
{
    uchar4*                image;
    unsigned int           image_width;
    unsigned int           image_height;
    float3                 cam_eye;
    float3                 cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
};

class PointQuery
{
    public:
    PointQuery();

    protected:
    void launchOptix();
    void createContext();
    void createModule();
    void createRaygenPrograms();
    void createMissPrograms();
    void createHitgroupPrograms();
    void createPipeline();
    void buildSBT();

    protected:
    //cuda stuff
    CUcontext          cudaContext;
    CUstream           stream;
    cudaDeviceProp     deviceProps;
    
    //optix context
    OptixDeviceContext optixContext;

    //optix pipeline
    OptixPipeline               pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions    pipelineLinkOptions    = {};

    //the module that contains our device programs
    OptixModule                 module;
    OptixModuleCompileOptions   moduleCompileOptions = {};

    //vectors of all our programs and the SBT built around them
    std::vector<OptixProgramGroup> raygenPGs;
    CUDABuffer raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    CUDABuffer missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    CUDABuffer hitgroupRecordsBuffer;
    OptixShaderBindingTable sbt = {};

    //launch param
    Params params;
    CUDABuffer   launchParamsBuffer;
    /*! @} */

    CUDABuffer colorBuffer;


};