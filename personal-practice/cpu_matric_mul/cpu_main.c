#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "cpu_matrix.h"
#include "cpu_io.h"

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static void print_usage(const char *prog_name) {
    printf("Usage: %s <input_file> [output_file]\n", prog_name);
    printf("  <input_file>   Path to input matrix file\n");
    printf("  [output_file]  Optional path to write output matrix\n");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    int n = 0;
    float *A = cpu_read_matrix(argv[1], &n);
    if (A == NULL) {
        return 1;
    }

    float *B = (float *)malloc((size_t)n * (size_t)n * sizeof(float));
    if (B == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for matrix B\n");
        free(A);
        return 1;
    }

    /* Initialize B as A * 2.0 for demonstration */
    for (int i = 0; i < n * n; i++) {
        B[i] = A[i] * 2.0f;
    }

    float *C = (float *)malloc((size_t)n * (size_t)n * sizeof(float));
    if (C == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for matrix C\n");
        free(A);
        free(B);
        return 1;
    }

    /* Start CPU timing */
    double cpu_time_start = get_time_ms();

    /* Perform matrix multiplication */
    cpu_matrix_multiply(A, B, C, n);

    /* End CPU timing */
    double cpu_time_end = get_time_ms();
    double cpu_time_total = cpu_time_end - cpu_time_start;

    /* Write output or print timing */
    if (argc == 3) {
        if (cpu_write_matrix(argv[2], C, n) != 0) {
            free(A);
            free(B);
            free(C);
            return 1;
        }
    }

    /* Print timing information */
    printf("CPU Matrix Multiplication Timing:\n");
    printf("  Matrix Size: %d x %d\n", n, n);
    printf("  Total Time: %.3f ms\n", cpu_time_total);
    printf("  Total Flops: %ld\n", (long)n * n * (2 * n - 1));

    /* Free host memory */
    free(A);
    free(B);
    free(C);

    return 0;
}
