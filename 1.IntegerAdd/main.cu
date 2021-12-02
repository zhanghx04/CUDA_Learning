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
