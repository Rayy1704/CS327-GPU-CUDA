#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "hpc.h"

const int block_dim=32;

typedef struct {
    int width;   /* Width of the image (in pixels) */
    int height;  /* Height of the image (in pixels) */
    int maxcol;  /* Largest color value (Used by the PPM read/write routines) */
    unsigned char *r, *g, *b; /* color channels (arrays of width x height elements each); each value must be less than or equal to maxcol */
} PPM_image;

/**
 * Read a PPM file from file `f`. This function is not very robust; it
 * may fail on perfectly legal PGM images, but works for the provided
 * cat.pgm file.
 */
void read_ppm( FILE *f, PPM_image* img )
{
    char buf[1024];
    const size_t BUFSIZE = sizeof(buf);
    char *s;
    int nread;

    assert(f != NULL);
    assert(img != NULL);

    /* Get the file type (must be "P6") */
    s = fgets(buf, BUFSIZE, f);
    if (0 != strcmp(s, "P6\n")) {
        fprintf(stderr, "FATAL: wrong file type %s\n", buf);
        exit(EXIT_FAILURE);
    }
    /* Get any comment and ignore it; does not work if there are
       leading spaces in the comment line */
    do {
        s = fgets(buf, BUFSIZE, f);
    } while (s[0] == '#');
    /* Get width, height */
    sscanf(s, "%d %d", &(img->width), &(img->height));
    /* get maxcol; must be less than or equal to 255 */
    s = fgets(buf, BUFSIZE, f);
    sscanf(s, "%d", &(img->maxcol));
    if ( img->maxcol > 255 ) {
        fprintf(stderr, "FATAL: maxcol=%d > 255\n", img->maxcol);
        exit(EXIT_FAILURE);
    }
    /* Get the binary data */
    img->r = (unsigned char*)malloc((img->width)*(img->height));
    assert(img->r != NULL);
    img->g = (unsigned char*)malloc((img->width)*(img->height));
    assert(img->g != NULL);
    img->b = (unsigned char*)malloc((img->width)*(img->height));
    assert(img->b != NULL);
    for (int k=0; k<(img->width)*(img->height); k++) {
        nread = fscanf(f, "%c%c%c", img->r + k, img->g + k, img->b + k);
        if (nread != 3) {
            fprintf(stderr, "FATAL: error reading pixel data\n");
            exit(EXIT_FAILURE);
        }
    }
}

/**
 * Write the image `img` to file `f`; is not NULL, use the string
 * `comment` as metadata.
 */
void write_ppm( FILE *f, const PPM_image* img, const char *comment )
{
    assert(f != NULL);
    assert(img != NULL);

    fprintf(f, "P6\n");
    fprintf(f, "# %s\n", comment != NULL ? comment : "");
    fprintf(f, "%d %d\n", img->width, img->height);
    fprintf(f, "%d\n", img->maxcol);
    for (int k=0; k<(img->width)*(img->height); k++) {
        fprintf(f, "%c%c%c", img->r[k], img->g[k], img->b[k]);
    }
}

/**
 * Free all memory used by the structure `img`
 */
void free_ppm( PPM_image* img )
{
    assert(img != NULL);
    free(img->r);
    free(img->g);
    free(img->b);
    img->r = img->g = img->b = NULL; /* not necessary */
    img->width = img->height = img->maxcol = -1;
}

/**
 * Swap *a and *b if necessary so that, at the end, *a <= *b
 */
__device__ void compare_and_swap( unsigned char *a, unsigned char *b )
{
    if (*a > *b ) {
        unsigned char tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

unsigned char *PTR(unsigned char *bmap, int width, int i, int j)
{
    return (bmap + i*width + j);
}

/**
 * Return the median of v[0..4]
 */
__device__ unsigned char median_of_five( unsigned char v[5] )
{
    /* We do a partial sort of v[5] using bubble sort until v[2] is
       correctly placed; this element is the median. (There are better
       ways to compute the median-of-five). */
    compare_and_swap( v+3, v+4 );
    compare_and_swap( v+2, v+3 );
    compare_and_swap( v+1, v+2 );
    compare_and_swap( v  , v+1 );
    compare_and_swap( v+3, v+4 );
    compare_and_swap( v+2, v+3 );
    compare_and_swap( v+1, v+2 );
    compare_and_swap( v+3, v+4 );
    compare_and_swap( v+2, v+3 );
    return v[2];
}

/**
 * Denoise a single color channel
 */

__global__ void denoise_kernel(unsigned char *in,unsigned char *out,int width, int height){
    __shared__ unsigned char tile[block_dim+2][block_dim+2];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * block_dim + tx;
    int row = blockIdx.y * block_dim + ty;
    int x = tx + 1;
    int y = ty + 1;

    if(row < height && col < width){
        tile[y][x] = in[row*width + col];
    }
    if(tx==0 && col>0){
        tile[y][0] = in[row*width + col-1];
    }
    if(tx==block_dim-1 && col<width-1){
        tile[y][block_dim+1] = in[row*width + col+1];
    }
    if(ty==0 && row>0){
        tile[0][x] = in[(row-1)*width + col];
    }
    if(ty==block_dim-1 && row<height-1){
        tile[block_dim+1][x] = in[(row+1)*width + col];
    }
    __syncthreads();

    if (row < height && col < width) {
        out[row*width + col] = in[row*width + col];
    }

    if (row >= 1 && row < height-1 && col >= 1 && col < width-1)
    {
        unsigned char temp[5];
        temp[0] = tile[y][x];
        temp[1] = tile[y][x-1];
        temp[2] = tile[y][x+1];
        temp[3] = tile[y-1][x];
        temp[4] = tile[y+1][x];

        out[row*width + col] = median_of_five(temp);
    }
}
void denoise(unsigned char *bmap, int width, int height){
    unsigned char *d_in, *d_out;
    size_t size = width * height * sizeof(unsigned char);
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, bmap, size, cudaMemcpyHostToDevice);

    dim3 block(block_dim, block_dim);
    dim3 grid((width + block_dim - 1) / block_dim, (height + block_dim - 1) / block_dim);
    denoise_kernel<<<grid, block>>>(d_in, d_out, width, height);
    cudaCheckError();

    cudaMemcpy(bmap, d_out, size, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}

int main( int argc, char* argv[] )
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.ppm> <output.ppm>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    PPM_image img;
    FILE *input_file = fopen(argv[1], "rb");
    if (input_file == NULL) {
        fprintf(stderr, "FATAL: could not open input file %s\n", argv[1]);
        exit(EXIT_FAILURE);
    }
    read_ppm(input_file, &img);
    fclose(input_file);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    denoise(img.r, img.width, img.height);
    denoise(img.g, img.width, img.height);
    denoise(img.b, img.width, img.height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    fprintf(stderr, "Execution time on GPU: %.3f ms\n", gpu_time);

    FILE *output_file = fopen(argv[2], "wb");
    if (output_file == NULL) {
        fprintf(stderr, "FATAL: could not open output file %s\n", argv[2]);
        exit(EXIT_FAILURE);
    }
    write_ppm(output_file, &img, "produced by cuda-denoise.cu");
    fclose(output_file);
    free_ppm(&img);
    return EXIT_SUCCESS;
}
