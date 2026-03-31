/*![Result of the Sobel operator](edge-detect.png)

The [Sobel operator](https://en.wikipedia.org/wiki/Sobel_operator) is
used to detect the edges on an grayscale image. The idea is to compute
the gradient of color change across each pixel; those pixels for which
the gradient exceeds a user-defined threshold are considered to be
part of an edge. Computation of the gradient involves the application
of a $3 \times 3$ stencil to the input image.

The program reads an input image fro standard input in
[PGM](https://en.wikipedia.org/wiki/Netpbm#PGM_example) (_Portable
Graymap_) format and produces a B/W image to standard output. The user
can specify an optional threshold on the command line.

The goal of this exercise is to parallelize the computation of the
Sobel operator using CUDA; this can be achieved by writing a kernel
that computes the edge at each pixel, and invoke the kernel from the
`edge_detect()` function.

To compile:

        nvcc cuda-edge-detect.cu -o cuda-edge-detect

To execute:

        ./cuda-edge-detect [threshold] < input > output

Example:

        ./cuda-edge-detect < BWstop-sign.pgm > BWstop-sign-edges.pgm

## Files

- [cuda-edge-detect.cu](cuda-edge-detect.cu) [hpc.h](hpc.h)
- [BWstop-sign.pgm](BWstop-sign.pgm)

***/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "hpc.h"

const int block_dim = 32;

typedef struct {
    int width;   /* Width of the image (in pixels) */
    int height;  /* Height of the image (in pixels) */
    int maxgrey; /* Don't care (used only by the PGM read/write routines) */
    unsigned char *bmap; /* buffer of width*height bytes; each element represents the gray level of a pixel (0-255) */
} PGM_image;

const unsigned char WHITE = 255;
const unsigned char BLACK = 0;

/**
 * Initialize a PGM_image object: allocate space for a bitmap of size
 * `width` x `height`, and set all pixels to color `col`
 */
void init_pgm( PGM_image *img, int width, int height, unsigned char col )
{
    int i, j;

    assert(img != NULL);

    img->width = width;
    img->height = height;
    img->maxgrey = 255;
    img->bmap = (unsigned char*)malloc(width*height);
    assert(img->bmap != NULL);
    for (i=0; i<height; i++) {
        for (j=0; j<width; j++) {
            img->bmap[i*width + j] = col;
        }
    }
}

/**
 * Read a PGM file from file `f`. Warning: this function is not
 * robust: it may fail on legal PGM images, and may crash on invalid
 * files since no proper error checking is done.
 */
void read_pgm( FILE *f, PGM_image* img )
{
    char buf[1024];
    const size_t BUFSIZE = sizeof(buf);
    char *s;
    int nread;

    assert(f != NULL);
    assert(img != NULL);

    /* Get the file type (must be "P5") */
    s = fgets(buf, BUFSIZE, f);
    if (0 != strcmp(s, "P5\n")) {
        fprintf(stderr, "Wrong file type %s\n", buf);
        exit(EXIT_FAILURE);
    }
    /* Get any comment and ignore it; does not work if there are
       leading spaces in the comment line */
    do {
        s = fgets(buf, BUFSIZE, f);
    } while (s[0] == '#');
    /* Get width, height */
    sscanf(s, "%d %d", &(img->width), &(img->height));
    /* get maxgrey; must be less than or equal to 255 */
    s = fgets(buf, BUFSIZE, f);
    sscanf(s, "%d", &(img->maxgrey));
    if ( img->maxgrey > 255 ) {
        fprintf(stderr, "FATAL: maxgray=%d > 255\n", img->maxgrey);
        exit(EXIT_FAILURE);
    }
#if _XOPEN_SOURCE < 600
    img->bmap = (unsigned char*)malloc((img->width)*(img->height)*sizeof(unsigned char));
#else
    /* The pointer img->bmap must be properly aligned to allow aligned
       SIMD load/stores to work. */
    int ret = posix_memalign((void**)&(img->bmap), __BIGGEST_ALIGNMENT__, (img->width)*(img->height));
    assert( 0 == ret );
#endif
    assert(img->bmap != NULL);
    /* Get the binary data from the file */
    nread = fread(img->bmap, 1, (img->width)*(img->height), f);
    if ( (img->width)*(img->height) != nread ) {
        fprintf(stderr, "FATAL: error reading input: expecting %d bytes, got %d\n", (img->width)*(img->height), nread);
        exit(EXIT_FAILURE);
    }
}

/**
 * Write the image `img` to file `f`; if not NULL, use the string
 * `comment` as metadata.
 */
void write_pgm( FILE *f, const PGM_image* img, const char *comment )
{
    assert(f != NULL);
    assert(img != NULL);

    fprintf(f, "P5\n");
    fprintf(f, "# %s\n", comment != NULL ? comment : "");
    fprintf(f, "%d %d\n", img->width, img->height);
    fprintf(f, "%d\n", img->maxgrey);
    fwrite(img->bmap, 1, (img->width)*(img->height), f);
}

/**
 * Free the bitmap associated with image `img`; note that the
 * structure pointed to by `img` is NOT deallocated; only `img->bmap`
 * is.
 */
void free_pgm( PGM_image *img )
{
    assert(img != NULL);
    free(img->bmap);
    img->bmap = NULL; /* not necessary */
    img->width = img->height = img->maxgrey = -1;
}

__host__ __device__ inline int IDX(int i, int j, int width)
{
    return (i*width + j);
}

__global__ void edge_detect_kernel(const unsigned char *in, unsigned char *out, int width, int height, int threshold)
{
    __shared__ unsigned char tile[block_dim + 2][block_dim + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int y = ty + 1;
    const int x = tx + 1;
    const int row = blockIdx.y * block_dim + ty;
    const int col = blockIdx.x * block_dim + tx;
    
   

    if (row < height && col < width) {
        tile[y][x] = in[IDX(row, col, width)];
    } else {
        tile[y][x] = 0;
    }

    if (tx == 0) {
        tile[y][0] = (row < height && col > 0) ? in[IDX(row, col - 1, width)] : 0;
    }
    if (tx == blockDim.x - 1) {
        tile[y][x + 1] = (row < height && col < width - 1) ? in[IDX(row, col + 1, width)] : 0;
    }
    if (ty == 0) {
        tile[0][x] = (col < width && row > 0) ? in[IDX(row - 1, col, width)] : 0;
    }
    if (ty == blockDim.y - 1) {
        tile[y + 1][x] = (col < width && row < height - 1) ? in[IDX(row + 1, col, width)] : 0;
    }

    if (tx == 0 && ty == 0) {
        tile[0][0] = (row > 0 && col > 0) ? in[IDX(row - 1, col - 1, width)] : 0;
    }
    if (tx == blockDim.x - 1 && ty == 0) {
        tile[0][x + 1] = (row > 0 && col < width - 1) ? in[IDX(row - 1, col + 1, width)] : 0;
    }
    if (tx == 0 && ty == blockDim.y - 1) {
        tile[y + 1][0] = (row < height - 1 && col > 0) ? in[IDX(row + 1, col - 1, width)] : 0;
    }
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
        tile[y + 1][x + 1] = (row < height - 1 && col < width - 1) ? in[IDX(row + 1, col + 1, width)] : 0;
    }

    __syncthreads();

    if (row >= height || col >= width) {
        return;
    }

    out[IDX(row, col, width)] = WHITE;

    if (row >= 1 && row < height - 1 && col >= 1 && col < width - 1) {
        const int gx =
            tile[y - 1][x - 1] - tile[y - 1][x + 1]
            + 2 * tile[y][x - 1] - 2 * tile[y][x + 1]
            + tile[y + 1][x - 1] - tile[y + 1][x + 1];
        const int gy =
            tile[y - 1][x - 1] + 2 * tile[y - 1][x] + tile[y - 1][x + 1]
            - tile[y + 1][x - 1] - 2 * tile[y + 1][x] - tile[y + 1][x + 1];
        const int magnitude = gx * gx + gy * gy;
        const int threshold_sq = threshold * threshold;
        out[IDX(row, col, width)] = (magnitude > threshold_sq) ? WHITE : BLACK;
    }
}


/**
 * Edge detection using the Sobel operator
 */
void edge_detect( const PGM_image* in, PGM_image* edges, int threshold )
{
    const int width = in->width;
    const int height = in->height;
    const size_t bytes = width * height * sizeof(unsigned char);
    unsigned char *d_in = NULL;
    unsigned char *d_out = NULL;

    cudaSafeCall(cudaMalloc(&d_in, bytes));
    cudaSafeCall(cudaMalloc(&d_out, bytes));
    cudaSafeCall(cudaMemcpy(d_in, in->bmap, bytes, cudaMemcpyHostToDevice));

    const dim3 block(block_dim, block_dim);
    const dim3 grid((width + block_dim - 1) / block_dim, (height + block_dim - 1) / block_dim);
    edge_detect_kernel<<<grid, block>>>(d_in, d_out, width, height, threshold);
    cudaCheckError();

    cudaSafeCall(cudaMemcpy(edges->bmap, d_out, bytes, cudaMemcpyDeviceToHost));

    cudaSafeCall(cudaFree(d_in));
    cudaSafeCall(cudaFree(d_out));
}

int main( int argc, char* argv[] )
{
    PGM_image bmap, out;
    int threshold = 70;

    if (argc != 3 && argc != 4) {
        fprintf(stderr, "Usage: %s <input.pgm> <output.pgm> [threshold]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc == 4) {
        threshold = atoi(argv[3]);
    }

    FILE *input_file = fopen(argv[1], "rb");
    if (input_file == NULL) {
        fprintf(stderr, "FATAL: could not open input file %s\n", argv[1]);
        return EXIT_FAILURE;
    }

    read_pgm(input_file, &bmap);
    fclose(input_file);

    init_pgm(&out, bmap.width, bmap.height, WHITE);

    cudaEvent_t start, stop;
    cudaSafeCall(cudaEventCreate(&start));
    cudaSafeCall(cudaEventCreate(&stop));
    cudaSafeCall(cudaEventRecord(start));

    edge_detect(&bmap, &out, threshold);

    cudaSafeCall(cudaEventRecord(stop));
    cudaSafeCall(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    cudaSafeCall(cudaEventElapsedTime(&elapsed_ms, start, stop));
    fprintf(stderr, "Execution time on GPU: %.3f ms\n", elapsed_ms);

    cudaSafeCall(cudaEventDestroy(start));
    cudaSafeCall(cudaEventDestroy(stop));

    FILE *output_file = fopen(argv[2], "wb");
    if (output_file == NULL) {
        fprintf(stderr, "FATAL: could not open output file %s\n", argv[2]);
        free_pgm(&bmap);
        free_pgm(&out);
        return EXIT_FAILURE;
    }

    write_pgm(output_file, &out, "produced by cuda-edge-detect-parallel.cu");
    fclose(output_file);

    free_pgm(&bmap);
    free_pgm(&out);
    return EXIT_SUCCESS;
}
