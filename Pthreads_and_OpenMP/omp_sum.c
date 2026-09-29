/*
 * Program: omp_sum.c
 * Purpose: OpenMP Work-Sharing Loop & Reduction
 * Description: Distributes array elements across threads and computes the total
 *              sum using the OpenMP reduction clause to prevent race conditions.
 */

#include <stdio.h>
#include <omp.h>

int main()
{
    int array[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    int total_sum = 0;

    // Parallel loop with reduction clause on total_sum
    #pragma omp parallel for reduction(+:total_sum)
    for (int i = 0; i < 8; i++)
    {
        int tid = omp_get_thread_num();
        printf("Thread %d processing array[%d] = %d\n", tid, i, array[i]);
        total_sum += array[i];
    }

    printf("Total sum = %d\n", total_sum);

    return 0;
}
