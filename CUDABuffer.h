#pragma once

// optix 7
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <assert.h>




#define CUDA_CHECK(call)							\
    {									\
      cudaError_t rc = cuda##call;                                      \
      if (rc != cudaSuccess) {                                          \
        std::stringstream txt;                                          \
        cudaError_t err =  rc; /*cudaGetLastError();*/                  \
        txt << "CUDA Error " << cudaGetErrorName(err)                   \
            << " (" << cudaGetErrorString(err) << ")";                  \
        throw std::runtime_error(txt.str());                            \
      }                                                                 \
    }

#define CUDA_CHECK_NOEXCEPT(call)                                        \
    {									\
      cuda##call;                                                       \
    }

#define OPTIX_CHECK( call )                                             \
  {                                                                     \
    OptixResult res = call;                                             \
    if( res != OPTIX_SUCCESS )                                          \
      {                                                                 \
        fprintf( stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__ ); \
        exit( 2 );                                                      \
      }                                                                 \
  }

#define CUDA_SYNC_CHECK()                                               \
  {                                                                     \
    cudaDeviceSynchronize();                                            \
    cudaError_t error = cudaGetLastError();                             \
    if( error != cudaSuccess )                                          \
      {                                                                 \
        fprintf( stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString( error ) ); \
        exit( 2 );                                                      \
      }                                                                 \
  }


struct CUDABuffer {
    inline CUdeviceptr d_pointer() const
    { return (CUdeviceptr)d_ptr; }

    //! re-size buffer to given number of bytes
    void resize(size_t size)
    {
      if (d_ptr) free();
      alloc(size);
    }
    
    //! allocate to given number of bytes
    void alloc(size_t size)
    {
      assert(d_ptr == nullptr);
      this->sizeInBytes = size;
      CUDA_CHECK(Malloc( (void**)&d_ptr, sizeInBytes));
    }

    //! free allocated memory
    void free()
    {
      CUDA_CHECK(Free(d_ptr));
      d_ptr = nullptr;
      sizeInBytes = 0;
    }

    template<typename T>
    void alloc_and_upload(const std::vector<T> &vt)
    {
      alloc(vt.size()*sizeof(T));
      upload((const T*)vt.data(),vt.size());
    }
    
    template<typename T>
    void upload(const T *t, size_t count)
    {
      assert(d_ptr != nullptr);
      assert(sizeInBytes == count*sizeof(T));
      CUDA_CHECK(Memcpy(d_ptr, (void *)t,
                        count*sizeof(T), cudaMemcpyHostToDevice));
    }
    
    template<typename T>
    void download(T *t, size_t count)
    {
      assert(d_ptr != nullptr);
      assert(sizeInBytes == count*sizeof(T));
      CUDA_CHECK(Memcpy((void *)t, d_ptr,
                        count*sizeof(T), cudaMemcpyDeviceToHost));
    }
    
    size_t sizeInBytes { 0 };
    void  *d_ptr { nullptr };
  };
