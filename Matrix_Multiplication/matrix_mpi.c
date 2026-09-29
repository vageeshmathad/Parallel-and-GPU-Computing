#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 4000

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = N / size;
    double *local_A = (double *)malloc(rows * N * sizeof(double));
    double *B = (double *)malloc(N * N * sizeof(double));
    double *local_C = (double *)calloc(rows * N, sizeof(double));
    double *A = NULL, *C = NULL;

    if (rank == 0) {
        A = (double *)malloc(N * N * sizeof(double));
        C = (double *)malloc(N * N * sizeof(double));
        for (int i = 0; i < N * N; i++) {
            A[i] = 1.0;
            B[i] = 1.0;
        }
    }

    double start = MPI_Wtime();

    MPI_Scatter(A, rows * N, MPI_DOUBLE, local_A, rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
            }
        }
    }

    MPI_Gather(local_C, rows * N, MPI_DOUBLE, C, rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0) {
        printf("Open MPI Distributed Matrix Multiplication Completed\n");
        printf("Matrix Size = %d x %d\n", N, N);
        printf("Number of Processes = %d\n", size);
        printf("Execution Time = %f seconds\n", end - start);
        printf("Verification C[0][0] = %.2f\n", C[0]);
        free(A);
        free(C);
    }

    free(local_A);
    free(B);
    free(local_C);
    MPI_Finalize();
    return 0;
}
