#ifndef CPU_IO_H
#define CPU_IO_H

/**
 * Reads a matrix from a file.
 * File format: first line contains n (size), followed by n*n floats.
 *
 * @param filename Path to input file
 * @param n Pointer to store matrix size
 * @return Pointer to allocated matrix, or NULL on error
 */
float* cpu_read_matrix(const char *filename, int *n);

/**
 * Writes a matrix to a file.
 *
 * @param filename Path to output file
 * @param matrix Matrix to write
 * @param n Size of matrix (n x n)
 * @return 0 on success, -1 on error
 */
int cpu_write_matrix(const char *filename, const float *matrix, int n);

#endif /* CPU_IO_H */
