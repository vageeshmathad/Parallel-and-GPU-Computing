#include <stdio.h>
#include <pthread.h>
#include <time.h>

#define N 1000000000L

double partial_sum[32];

typedef struct
{
    int thread_id;
    long start;
    long end;
} ThreadData;

double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

void *calculate(void *arg)
{
    ThreadData *data = (ThreadData *)arg;
    double sum = 0.0;

    for (long i = data->start; i < data->end; i++)
    {
        sum += (double)i * 0.000001;
    }

    partial_sum[data->thread_id] = sum;
    return NULL;
}

int main()
{
    int num_threads;
    printf("Enter number of threads: ");
    scanf("%d", &num_threads);

    if (num_threads < 1 || num_threads > 32)
    {
        printf("Please enter a value between 1 and 32.\n");
        return 1;
    }

    pthread_t threads[num_threads];
    ThreadData data[num_threads];
    long chunk = N / num_threads;

    double start_time = get_time();

    for (int i = 0; i < num_threads; i++)
    {
        data[i].thread_id = i;
        data[i].start = i * chunk;
        if (i == num_threads - 1)
            data[i].end = N;
        else
            data[i].end = (i + 1) * chunk;

        pthread_create(
            &threads[i],
            NULL,
            calculate,
            &data[i]
        );
    }

    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    double total_sum = 0.0;
    for (int i = 0; i < num_threads; i++)
    {
        total_sum += partial_sum[i];
    }

    double end_time = get_time();

    printf("Result = %.2f\n", total_sum);
    printf("Execution time = %.6f seconds\n", end_time - start_time);
    return 0;
}
