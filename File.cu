#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

__global__ void test(int* a, int* b, int n) {
    int i = threadIdx.x;

    if (i < n) {
        a[i] += b[i];
    }
}
__global__ void MatrixMultiply(int *a, int *b, int n) {



}
/*
int main() {
    int n = 2;
    int* a;
    int* b;

    // Allocate unified memory for arrays a and b
    cudaError_t err = cudaMallocManaged(&a, n * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMallocManaged failed for a: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    err = cudaMallocManaged(&b, n * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMallocManaged failed for b: " << cudaGetErrorString(err) << std::endl;
        cudaFree(a);
        return -1;
    }

    // Initialize arrays
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i + 1;
    }

    // Launch kernel with enough threads to cover all elements
    test << <1, n >> > (a, b, n);

    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(a);
        cudaFree(b);
        return -1;
    }

    // Print results
    for (int i = 0; i < n; i++) {
        std::cout << a[i] << std::endl;
    }

    // Free allocated memory
    cudaFree(a);
    cudaFree(b);

    return 0;
}
/