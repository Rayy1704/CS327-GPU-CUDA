#include "cpu_io.h"
#include <stdio.h>
#include <stdlib.h>

float* cpu_read_matrix(const char *filename, int *n) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }

    if (fscanf(fp, "%d", n) != 1) {
        fprintf(stderr, "Error: Cannot read matrix size\n");
        fclose(fp);
        return NULL;
    }

    if (*n <= 0) {
        fprintf(stderr, "Error: Invalid matrix size %d\n", *n);
        fclose(fp);
        return NULL;
    }

    float *matrix = (float *)malloc((size_t)*n * (size_t)*n * sizeof(float));
    if (matrix == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for matrix of size %d\n", *n);
        fclose(fp);
        return NULL;
    }

    for (int i = 0; i < *n * *n; i++) {
        if (fscanf(fp, "%f", &matrix[i]) != 1) {
            fprintf(stderr, "Error: Cannot read matrix element at index %d\n", i);
            free(matrix);
            fclose(fp);
            return NULL;
        }
    }

    fclose(fp);
    return matrix;
}

int cpu_write_matrix(const char *filename, const float *matrix, int n) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
        return -1;
    }

    if (fprintf(fp, "%d\n", n) < 0) {
        fprintf(stderr, "Error: Cannot write matrix size\n");
        fclose(fp);
        return -1;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fprintf(fp, "%.2f ", matrix[i * n + j]) < 0) {
                fprintf(stderr, "Error: Cannot write matrix element\n");
                fclose(fp);
                return -1;
            }
        }
        if (fprintf(fp, "\n") < 0) {
            fprintf(stderr, "Error: Cannot write newline\n");
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);
    return 0;
}
