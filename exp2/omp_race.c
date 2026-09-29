/*
 * Program: omp_race.c
 * Purpose: OpenMP Race Condition Demonstration
 * Description: Multiple threads concurrently increment a shared variable without
 *              synchronization, causing non-atomic lost updates and data race.
 */

#include <stdio.h>
#include <omp.h>

int main()
{
    int counter = 0;
    int expected = 400000;

    // 4 threads concurrently execute 100,000 increments each
    #pragma omp parallel num_threads(4)
    {
        for (int i = 0; i < 100000; i++)
        {
            counter++; // Race condition: non-atomic read-modify-write
        }
    }

    printf("Expected counter = %d\n", expected);
    printf("Actual counter   = %d\n", counter);

    return 0;
}
