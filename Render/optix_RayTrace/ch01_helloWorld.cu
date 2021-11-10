
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <stdio.h>
#include <iostream>

#define OPTIX_CHECK( call )                                             \
  {                                                                     \
    OptixResult res = call;                                             \
    if( res != OPTIX_SUCCESS )                                          \
      {                                                                 \
        fprintf( stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__ ); \
        exit( 2 );                                                      \
      }                                                                 \
  }

void initOptix()
{
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
        throw std::runtime_error("no CUDA capable devices found !");
    std::cout << "found " << numDevices << " cuda devices" << std::endl;
    OPTIX_CHECK(optixInit());
}

int main()
{

    try
    {
        std::cout << "initializing optix..." << std::endl;
        initOptix();
    }
    catch (std::runtime_error& e)
    {
        std::cout << " FATAL ERROR:" << e.what() << std::endl;
        exit(1);
    }
    return 0;

}

