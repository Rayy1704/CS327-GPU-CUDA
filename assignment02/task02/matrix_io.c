#include "matrix_io.h"
#include <stdlib.h>

Matrix read_matrix_from_file(FILE *fp) {
    Matrix m = {0, 0, NULL};
    if (fscanf(fp, "%d %d", &m.rows, &m.cols) != 2) return m;

    m.data = (float *)malloc(m.rows * m.cols * sizeof(float));
    if(!m.data) return m; //allocation failed
    for (int i = 0; i < m.rows * m.cols; i++) {
        if (fscanf(fp, "%f", &m.data[i]) != 1) {
            fprintf(stderr, "Error: Failed to read matrix data at index %d\n", i);
            free(m.data);
            m.data = NULL;
            return m;
        }
    }
    return m;
}

void write_matrix_to_file(FILE *fp, Matrix m) {
    fprintf(fp, "%d %d\n", m.rows, m.cols);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            fprintf(fp, "%.2f%s", m.data[i * m.cols + j], (j == m.cols - 1 ? "" : " "));
        }
        fprintf(fp, "\n");
    }
}

void free_matrix(Matrix m) {
    if (m.data) free(m.data);
}
