#ifndef MATRIX_IO_H
#define MATRIX_IO_H
#include "matrix.h"
#include <stdio.h>


Matrix read_matrix_from_file(FILE *fp);
void write_matrix_to_file(FILE *fp, Matrix m);
void free_matrix(Matrix m);

#endif