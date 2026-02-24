#include "matrix.h"
#include<stdlib.h>

Matrix multiply_matrices(Matrix A, Matrix B) {
    Matrix C = {A.rows, B.cols, (float *)calloc(A.rows * B.cols, sizeof(float))};
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < B.cols; j++) {
            for (int k = 0; k < A.cols; k++) {
                C.data[i * C.cols + j] += A.data[i * A.cols + k] * B.data[k * B.cols + j];
            }
        }
    }
    return C;
}