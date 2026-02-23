#ifndef GPU_MATRIX_H
#define GPU_MATRIX_H

/**
 * GPU-accelerated matrix multiplication kernel.
 * Multiplies two square matrices A and B, storing result in C.
 * All matrices are n x n.
 *
 * @param d_A Device pointer to first input matrix (row-major order)
 * @param d_B Device pointer to second input matrix (row-major order)
 * @param d_C Device pointer to output matrix (row-major order)
 * @param n Size of matrices (n x n)
 */
void gpu_matrix_multiply(float *d_A, float *d_B, float *d_C, int n);

/**
 * Allocates GPU memory and copies host data to device.
 *
 * @param h_data Host data pointer
 * @param size Number of elements
 * @return Device pointer
 */
float* gpu_malloc_and_copy(const float *h_data, size_t size);

/**
 * Frees GPU memory.
 *
 * @param d_ptr Device pointer
 */
void gpu_free(float *d_ptr);

#endif /* GPU_MATRIX_H */
