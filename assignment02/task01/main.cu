#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("--- Task 1: CUDA Device Properties ---\n\n");
    printf("prop.name: The model name of the GPU. Value = %s\n", prop.name);
    printf("prop.totalGlobalMem: Total VRAM capacity. Value = %zu bytes (~%zu MB)\n", prop.totalGlobalMem, prop.totalGlobalMem / (1024 * 1024));
    printf("prop.sharedMemPerBlock: Shared memory per block. Value = %zu bytes\n", prop.sharedMemPerBlock);
    printf("prop.regsPerBlock: Number of 32-bit registers per block. Value = %d\n", prop.regsPerBlock);
    printf("prop.warpSize: Number of threads that execute in SIMD architecture. Value = %d\n", prop.warpSize);
    printf("prop.maxThreadsPerBlock: Maximum threads allowed per block block. Value = %d\n", prop.maxThreadsPerBlock);
    printf("prop.clockRate: The maximum clock speed. Value = %.2f GHz\n", (float)prop.clockRate / 1e6);
    printf("prop.multiProcessorCount: Total number of SM. Value = %d\n", prop.multiProcessorCount);
    printf("prop.memoryClockRate: The maximum memory clock speed. Value = %.2f GHz\n", (float)prop.memoryClockRate / 1e6);
    printf("prop.memoryBusWidth: The width of the memory interface(bits). Value = %d-bit\n", prop.memoryBusWidth);
    printf("prop.major.minor: The Compute Capability version. Value = %d.%d\n", prop.major, prop.minor);

    printf("\n--- Performance Calculations ---\n");
    
    //calculations for max global memory bandwidth
    //formula= (memoryClock * 2 [DDR] * (busWidth/ 8))/10^9
    double bandwidth = (2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8.0)) / 1e6;
    printf("Max Global Memory Bandwidth: %.2f GB/s\n", bandwidth);

   int coresPerSM = 0;
    if (prop.major == 8) {
        coresPerSM = 128; 
    } else if (prop.major == 7 && prop.minor == 5) {
        coresPerSM = 64; // Turing
    } else {
        coresPerSM = 128; // Default for most modern cards
    }

    double peakGFLOPS = ((double)prop.clockRate * prop.multiProcessorCount * coresPerSM * 2.0) / 1.0e6;

    printf("Peak Compute Performance for %s: %.2f GFLOPS\n", prop.name, peakGFLOPS);

    return 0;
}