#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int rows;
    int cols;
    float *data; 
} Matrix;

Matrix create_matrix(int rows, int cols);
#endif