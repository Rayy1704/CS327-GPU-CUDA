#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
__global__
void vecAddKernel(float *A_h,float *B_h,float *C_h,int n){
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i<n){
        C_h[i]=A_h[i]+B_h[i];
    }
}
static void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

void vecAdd(float *A_h,float *B_h,float *C_h,int n){
    float *A_d = NULL, *B_d = NULL, *C_d = NULL;
    size_t size = (size_t)n * sizeof(float);
    int block_size = 256, grid_size = (n + block_size - 1) / block_size;
    cudaError_t err;

    err = cudaMalloc((void**)&A_d, size);
    checkCuda(err, "cudaMalloc A_d failed");
    err = cudaMalloc((void**)&B_d, size);
    checkCuda(err, "cudaMalloc B_d failed");
    err = cudaMalloc((void**)&C_d, size);
    checkCuda(err, "cudaMalloc C_d failed");

    err = cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    checkCuda(err, "cudaMemcpy A_d failed");
    err = cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    checkCuda(err, "cudaMemcpy B_d failed");

    vecAddKernel<<<grid_size,block_size>>>(A_d,B_d,C_d,n);
    err = cudaGetLastError();
    checkCuda(err, "kernel launch failed");
    err = cudaDeviceSynchronize();
    checkCuda(err, "kernel execution failed");

    err = cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    checkCuda(err, "cudaMemcpy C_h failed");

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(int argc, char* argv[]){
    int n=1000;
    float * A,*B,*C;
    A=(float*)malloc((size_t)n*sizeof(float));
    B=(float*)malloc((size_t)n*sizeof(float));
    C=(float*)malloc((size_t)n*sizeof(float));
    if (!A || !B || !C) {
        fprintf(stderr, "host allocation failed\n");
        return EXIT_FAILURE;
    }
    for(int i=0;i<n;i++){
        A[i]=i;
        B[i]=i*2;
    }
    vecAdd(A,B,C,n);
    for(int i=990;i<1000 && i<n;i++){
        printf("C[%d] = %f\n", i, C[i]);
    }
    printf("\n");

    free(A);
    free(B);
    free(C);

    cudaDeviceReset();
    return 0;
}