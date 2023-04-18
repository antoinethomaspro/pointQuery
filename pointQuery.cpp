#include <iostream>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include "pointQuery.h"

#include <sutil/sutil.h>





 /*! SBT record for a raygen program */
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
{
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  // just a dummy value - later examples will use more interesting
  // data here
  void *data;
};

/*! SBT record for a miss program */
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
{
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  // just a dummy value - later examples will use more interesting
  // data here
  void *data;
};

/*! SBT record for a hitgroup program */
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
{
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  // just a dummy value - later examples will use more interesting
  // data here
  int objectID;
};



PointQuery::PointQuery()
{
    launchOptix();
}

void PointQuery::launchOptix()
{
    std::cout << "#osc: setting up module ..." << std::endl;
    optixInit();
    std::cout << "#osc: setting up module ..." << std::endl;

}

static void context_log_cb(unsigned int level,
                             const char *tag,
                             const char *message,
                             void *)
  {
    fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
  }

void PointQuery::createContext()
{    // for this sample, do everything on one device !! error to check
    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));
    CUDA_CHECK(StreamCreate(&stream));
      
    cudaGetDeviceProperties(&deviceProps, deviceID);
    std::cout << "#osc: running on device: " << deviceProps.name << std::endl;
      
    CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
    if( cuRes != CUDA_SUCCESS ) 
      fprintf( stderr, "Error querying current context: error code %d\n", cuRes );
      
    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback
                (optixContext,context_log_cb,nullptr,4));

}




void PointQuery::createModule()
{
  moduleCompileOptions.maxRegisterCount  = 50;
  moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  pipelineCompileOptions = {};
  pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipelineCompileOptions.usesMotionBlur     = false;
  pipelineCompileOptions.numPayloadValues   = 2;
  pipelineCompileOptions.numAttributeValues = 2;
  pipelineCompileOptions.exceptionFlags     = OPTIX_EXCEPTION_FLAG_NONE;
  pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams"; // a mettre en relation avec le params dans cu

  pipelineLinkOptions.maxTraceDepth          = 2;
    
   size_t      inputSize = 0;
    const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixInOneWeekend.cu", inputSize);

    // Create Module from PTX
    char   log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixModuleCreateFromPTX(
                                              optixContext,
                                              &moduleCompileOptions,
                                              &pipelineCompileOptions,
                                              input,
                                              inputSize,
                                              log,
                                              &sizeof_log,
                                              &module
                                          ));
  
  
}

void PointQuery::createRaygenPrograms()
{

}

void PointQuery::createMissPrograms()
{
  
}

void PointQuery::createHitgroupPrograms()
{

}

void PointQuery::createPipeline()
{

}

void PointQuery::buildSBT()
{
  
}










/*! main entry point to this example - initially optix, print hello
  world, then exit */
extern "C" int main(int ac, char **av)
{
  
  PointQuery query;
  


  return 0;
}




