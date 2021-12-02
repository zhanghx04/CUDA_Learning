# This Part is the Introduction with a simple CUDA example

## How to Run
> cd 1.IntegerAdd
>
> mkdir build && cd build
> 
> cmake ..
>
> make
>
> ./simple_example


## CMakeLists.txt

Here I will explain each steps in __CMakeLists.txt__ file

---

This is to tell  the minimum cmake version is required
> cmake_minimum_required(VERSION 2.8.12)

Tell the Project's Name. Could simply say PROJECT(simple_example) 
> SET(PROJECTNAME simple_example)
>
> PROJECT(${PROJECTNAME})

Tell what file need to be compile
> SET(PROJECT_SRCS
>     main.cu
> ) 


Tell the C++ version
> SET(CMAKE_CXX_FLAGS "-std=c++11")

Find CUDA
> FIND_PACKAGE(CUDA REQUIRED)
>
> INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

Set CUDA libraries
> SET(LIBRARIES ${CUDA_LIBRARIES})

Set exeutable file
> CUDA_ADD_EXECUTABLE(\${PROJECTNAME} ${PROJECT_SRCS})

Link the libraries to the project
> TARGET_LINK_LIBRARIES(${PROJECTNAME} ${LIBRARIES})


## CUDA Example
``` cpp
#include <iostream>
#include <cuda.h>

using namespace std;

__global__ void AddIntsCUDA(int* a, int* b)
{
    a[0] += b[0];
}

int main()
{
    int a = 5;
    int b = 9;
    
    int *d_a, *d_b;
    
    // Malloc space for device variables
    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));
    
    // Copy memory to device
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
    
    AddIntsCUDA<<<1, 1>>>(d_a, d_b);
    
    // copy the value from device to host back
    cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost);
    
    cout << "The answer is " << a << endl;
    
    // Release the memory
    cudaFree(d_a);
    cudaFree(d_b);
    
    return 0;
}
```