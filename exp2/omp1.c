/*
 * Program: omp1.c
 * Purpose: OpenMP Thread Creation & Identification
 * Description: Spawns a parallel team of 16 threads and queries thread ID
 *              and team size using OpenMP runtime library functions.
 */

#include <stdio.h>
#include <omp.h>

int main()
{
    // Fork a team of 16 threads
    #pragma omp parallel num_threads(16)
    {
        int tid = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        printf("Hello from Thread %d of %d\n", tid, total_threads);
    }

    return 0;
}
