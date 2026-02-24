#include "matrix_io.h"
#include "matrix_ops.h"
#include "ci.h"
#include "matrix.h"
#include <cuda_runtime.h>


int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <input_file> [output_file]\n", argv[0]);
        return 1;
    }

    FILE *in = fopen(argv[1], "r");
    if (!in) { perror("Input error"); return 1; }

    Matrix A = read_matrix_from_file(in);
    Matrix B = read_matrix_from_file(in);
    fclose(in);

    if (A.cols != B.rows) {
        fprintf(stderr, "Dimension mismatch!\n");
        return 1;
    }
    Matrix C = create_matrix(A.rows, B.cols);
    multiply_matrices_cuda(A, B, &C);
    // append_to_csv("cudaTime.csv", A.rows, A.cols, B.rows, B.cols, time);
    FILE *out = (argc == 3) ? fopen(argv[2], "w") : stdout;//if ouput path is'nt given direct to stdout
    write_matrix_to_file(out, C);
    if (argc == 3) fclose(out);
    print_computational_intensity(A, B);
    free_matrix(A); free_matrix(B); free_matrix(C);
    return 0;
}