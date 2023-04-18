#pragma once

#include <vector>
#include "CUDABuffer.h"


struct Params
{
    int       frameID { 0 };
    uint32_t *colorBuffer;
    float2 fbSize;
};

class PointQuery
{
    public:
    PointQuery();

    void render();

    void resize(const float2 &newSize);

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