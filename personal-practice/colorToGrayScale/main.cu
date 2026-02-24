#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<ctype.h>
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

static int read_next_int(FILE *fp, int *out){
    int c;
    do {
        c = fgetc(fp);
        if (c == '#') {
            while (c != '\n' && c != EOF) {
                c = fgetc(fp);
            }
        }
    } while (isspace(c));

    if (c == EOF) {
        return 0;
    }
    ungetc(c, fp);
    return fscanf(fp, "%d", out) == 1;
}

static int read_ppm(const char *path, unsigned char **data, int *width, int *height){
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        return 0;
    }

    char magic[3] = {0};
    if (fscanf(fp, "%2s", magic) != 1) {
        fclose(fp);
        return 0;
    }

    int w = 0;
    int h = 0;
    int maxval = 0;
    if (!read_next_int(fp, &w) || !read_next_int(fp, &h) || !read_next_int(fp, &maxval)) {
        fclose(fp);
        return 0;
    }
    if (w <= 0 || h <= 0 || maxval <= 0) {
        fclose(fp);
        return 0;
    }

    size_t size = (size_t)w * (size_t)h * 3;
    unsigned char *pixels = (unsigned char *)malloc(size);
    if (!pixels) {
        fclose(fp);
        return 0;
    }

    if (strcmp(magic, "P6") == 0) {
        fgetc(fp);
        if (fread(pixels, 1, size, fp) != size) {
            free(pixels);
            fclose(fp);
            return 0;
        }
    } else if (strcmp(magic, "P3") == 0) {
        for (size_t i = 0; i < size; i++) {
            int value = 0;
            if (!read_next_int(fp, &value)) {
                free(pixels);
                fclose(fp);
                return 0;
            }
            if (maxval != 255) {
                value = (value * 255) / maxval;
            }
            if (value < 0) value = 0;
            if (value > 255) value = 255;
            pixels[i] = (unsigned char)value;
        }
    } else {
        free(pixels);
        fclose(fp);
        return 0;
    }

    fclose(fp);
    *data = pixels;
    *width = w;
    *height = h;
    return 1;
}

static int write_pgm(const char *path, const unsigned char *data, int width, int height){
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        return 0;
    }
    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    size_t size = (size_t)width * (size_t)height;
    int ok = fwrite(data, 1, size, fp) == size;
    fclose(fp);
    return ok;
}

int main(int argc, char **argv){
    const char *inputPath = "sample.ppm";
    const char *outputPath = "out.pgm";
    if (argc >= 3) {
        inputPath = argv[1];
        outputPath = argv[2];
    }

    unsigned char *h_Pin = NULL;
    int width = 0;
    int height = 0;
    if (!read_ppm(inputPath, &h_Pin, &width, &height)) {
        fprintf(stderr, "Failed to read %s\n", inputPath);
        return 1;
    }

    size_t graySize = (size_t)width * (size_t)height;
    unsigned char *h_Pout = (unsigned char *)malloc(graySize);
    if (!h_Pout) {
        free(h_Pin);
        return 1;
    }

    colorToGrayScale(h_Pin, h_Pout, width, height);

    if (!write_pgm(outputPath, h_Pout, width, height)) {
        fprintf(stderr, "Failed to write %s\n", outputPath);
        free(h_Pin);
        free(h_Pout);
        return 1;
    }

    free(h_Pin);
    free(h_Pout);
    cudaDeviceReset();
    return 0;
}
