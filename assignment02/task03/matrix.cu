#include "matrix.h"

Matrix create_matrix(int rows, int cols) {
    Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.data = (float*)calloc(rows * cols , sizeof(float));
    return mat;
}