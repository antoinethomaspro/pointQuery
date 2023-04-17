#include <iostream>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include "pointQuery.h"




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
      
    


  }





/*! main entry point to this example - initially optix, print hello
  world, then exit */
extern "C" int main(int ac, char **av)
{
  
    std::cout << "#initializing optix..." << std::endl;
    
    optixInit();
    
    std::cout << "#done!" << std::endl;

  return 0;
}




