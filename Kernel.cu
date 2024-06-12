#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// Preallocated memory for result

__global__ void dot(int a_x, int a_y, int b_x, int b_y, int* result) {
    *result = a_x * b_x + a_y * b_y;
}
__global__ void add(int a, int b, int* result) {
    *result = a + b;
}

int dotProduct(int a_x, int a_y, int b_x, int b_y) {
    int result; 
    int* d_result = nullptr;

    cudaMallocManaged(&d_result, 4);

    dot << <1, 256 >> > (a_x, a_y, b_x, b_y, d_result);
    //cudaDeviceSynchronize();

    cudaMemcpy(&result, d_result, 4, cudaMemcpyDeviceToHost);

    //cudaFree(d_result); // Free memory immediately after use
    cudaFree(d_result);
    return result;
    //cudaFree(&result);
}


int addNum(int a, int b) {
    int result;

    int* d_result = nullptr;

    cudaMallocManaged(&d_result, 4);

    add << <1, 256 >> > (a,b, d_result);

    cudaMemcpy(&result, d_result, 4, cudaMemcpyDeviceToHost);

    //cudaDeviceSynchronize();

    cudaFree(d_result);

    return result;
    //cudaFree(&result);

}
