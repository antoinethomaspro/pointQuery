#include <iostream>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>


  
  /*! main entry point to this example - initially optix, print hello
    world, then exit */
  extern "C" int main(int ac, char **av)
  {
    
      std::cout << "#initializing optix..." << std::endl;
      
      optixInit();
      
      std::cout << "#done!" << std::endl;

    return 0;
  }




