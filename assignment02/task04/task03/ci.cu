#include <stdio.h>
#include "matrix.h"

void print_computational_intensity(Matrix A, Matrix B) {
    long long M = (long long)A.rows;
    long long N = (long long)A.cols;
    long long P = (long long)B.cols;

    double total_ops = 2.0 * M * N * P;

    double total_access = 2*M*N*P;

    double intensity = total_ops / total_access;

    printf("Comp. Intensity:   %.4f FLOPs/access\n", intensity);
    printf("Total FLOPs:       %.2e\n", total_ops);
}