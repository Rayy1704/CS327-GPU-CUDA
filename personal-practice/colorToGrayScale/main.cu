#include<stdio.h>
#include<cuda_runtime.h>
static void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
__global__
void colorToGrayScaleKernel(unsigned char* Pout,unsigned char * Pin , int width , int height){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if(col<width && row<height){
        int grayOffset = row*width+col;
        int rgbOffset = grayOffset*3;
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset+1];     
        unsigned char b = Pin[rgbOffset+2];
        Pout[grayOffset]= 0.21*r+0.72*g+0.07f*b;
    }
}

void colorToGrayScale(unsigned char* Pin, unsigned char* Pout, int width, int height){
    unsigned char * d_Pin=NULL,*d_Pout=NULL;
    size_t size = (size_t)(width*height)*sizeof(unsigned char)*3;
    cudaError_t err;
    err=cudaMalloc(&d_Pin,size);
    checkCuda(err,"cudaMalloc d_Pin");
    err=cudaMalloc(&d_Pout,size/3);
    checkCuda(err,"cudaMalloc d_Pout");
    err=cudaMemcpy(d_Pin,Pin,size,cudaMemcpyHostToDevice);
    checkCuda(err,"cudaMemcpy d_Pin");
    dim3 blockSize(16,16);
    dim3 gridSize((width+blockSize.x-1)/blockSize.x,(height+blockSize.y-1)/blockSize.y);
    colorToGrayScaleKernel<<<gridSize,blockSize>>>(d_Pout,d_Pin,width,height);
    err=cudaGetLastError();
    checkCuda(err,"kernel launch failed");
    err=cudaDeviceSynchronize();
    checkCuda(err,"kernel execution failed");
    err=cudaMemcpy(Pout,d_Pout,size/3,cudaMemcpyDeviceToHost);
    checkCuda(err,"cudaMemcpy d_Pout");
    cudaFree(d_Pin);
    cudaFree(d_Pout);
}
int main(){
    int width = 1920;
    int height=1080;
    size_t size =(size_t)(width*height)*sizeof(unsigned char)*3;
    unsigned char * h_Pin = (unsigned char*)malloc(size);
    unsigned char * h_Pout = (unsigned char*)malloc(size/3);
    if(!h_Pin || !h_Pout){
        fprintf(stderr,"host allocation failed\n");
        return EXIT_FAILURE;
    }
    for(int i=0;i<width*height*3;i++){
        h_Pin[i]=rand()%256;
    }
    colorToGrayScale(h_Pin,h_Pout,width,height);
    free(h_Pin);
    free(h_Pout);
    cuda_DeviceReset();
    return 0;
}
