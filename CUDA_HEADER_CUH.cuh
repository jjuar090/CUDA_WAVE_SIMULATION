// CUDA_HEADER_CUH.cuh
#ifndef CUDA_HEADER_CUH
#define CUDA_HEADER_CUH

int* d_result;


void Clear(); 
void dot(int a, int b, int c, int d, int* e);
void add(int a, int b);

void initCUDA();
void cleanupCUDA();




int addNum(int a, int b);
int dotProduct(int a, int b, int c, int d);

#endif  // CUDA_HEADER_CUH
