/*
 * Program: omp_barrier.c
 * Purpose: OpenMP Barrier Synchronization
 * Description: Demonstrates phase-based coordination using #pragma omp barrier,
 *              ensuring all threads finish Stage 1 before any thread enters Stage 2.
 */

#include <stdio.h>
#include <omp.h>

int main()
{
    // 4 threads executing in phased synchronization
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();

        // Stage 1
        printf("Thread %d completed Stage 1\n", tid);

        // Synchronization point: all threads must arrive here before proceeding
        #pragma omp barrier

        // Stage 2
        printf("Thread %d started Stage 2\n", tid);
    }

    return 0;
}
