/*
 * Program: omp_critical.c
 * Purpose: OpenMP Critical Section for Mutual Exclusion
 * Description: Eliminates race conditions by enforcing mutual exclusion
 *              using the #pragma omp critical directive.
 */

#include <stdio.h>
#include <omp.h>

int main()
{
    int counter = 0;
    int expected = 400000;

    // 4 threads execute 100,000 increments each with mutual exclusion
    #pragma omp parallel num_threads(4)
    {
        for (int i = 0; i < 100000; i++)
        {
            #pragma omp critical
            {
                counter++; // Critical section guarantees atomic update
            }
        }
    }

    printf("Expected counter = %d\n", expected);
    printf("Actual counter   = %d\n", counter);

    return 0;
}
