#include <stdio.h>
#include <time.h>

#define N 1000000000L

double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main()
{
    double sum = 0.0;
    double start = get_time();

    for (long i = 0; i < N; i++)
    {
        sum += (double)i * 0.000001;
    }

    double end = get_time();

    printf("Result = %.2f\n", sum);
    printf("Execution time = %.6f seconds\n", end - start);
    return 0;
}
