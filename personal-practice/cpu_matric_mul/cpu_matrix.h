#ifndef CPU_MATRIX_H
#define CPU_MATRIX_H

/**
 * Multiplies two square matrices A and B, storing result in C.
 * All matrices are n x n.
 *
 * @param A First input matrix (row-major order)
 * @param B Second input matrix (row-major order)
 * @param C Output matrix (row-major order)
 * @param n Size of matrices (n x n)
 */
void cpu_matrix_multiply(float *A, float *B, float *C, int n);

#endif /* CPU_MATRIX_H */
