#include "matrix.h"
#include<stdlib.h>
#include <stdio.h>
static void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
__global__ void matMulKernel(float* A, float* B, float* C, int hA, int wA, int wB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < hA && col < wB) {
        float sum = 0.0f;
        for (int k = 0; k < wA; k++) {
            sum += A[row * wA + k] * B[k * wB + col];
        }
        C[row * wB + col] = sum;
    }
}
float multiply_matrices_cuda(Matrix A, Matrix B, Matrix *C) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    float *d_A, *d_B, *d_C;
    size_t sizeA = A.rows * A.cols * sizeof(float);
    size_t sizeB = B.rows * B.cols * sizeof(float);
    size_t sizeC = A.rows * B.cols * sizeof(float);
    cudaError_t err;
    err =cudaMalloc(&d_A, sizeA);
    checkCuda(err, "Failed to allocate device memory for A");
    err = cudaMalloc(&d_B, sizeB);
    checkCuda(err, "Failed to allocate device memory for B");
    err = cudaMalloc(&d_C, sizeC);
    checkCuda(err, "Failed to allocate device memory for C");


    err= cudaMemcpy(d_A, A.data, sizeA, cudaMemcpyHostToDevice);
    checkCuda(err, "Failed to copy A to device");
    err = cudaMemcpy(d_B, B.data, sizeB, cudaMemcpyHostToDevice);
    checkCuda(err, "Failed to copy B to device");

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((B.cols + 15) / 16, (A.rows + 15) / 16);

    matMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, A.rows, A.cols, B.cols);
    err = cudaGetLastError();
    checkCuda(err, "Kernel launch failed"); 
    err=cudaDeviceSynchronize();
    checkCuda(err, "Kernel execution failed");

    err=cudaMemcpy(C->data, d_C, sizeC, cudaMemcpyDeviceToHost);
    checkCuda(err, "Failed to copy C to host");

  
    err= cudaFree(d_A);
    checkCuda(err, "Failed to free device memory for A");
    err = cudaFree(d_B);
    checkCuda(err, "Failed to free device memory for B");
    err = cudaFree(d_C);
    checkCuda(err, "Failed to free device memory for C");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float te = 0;
    cudaEventElapsedTime(&te, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return te;
}