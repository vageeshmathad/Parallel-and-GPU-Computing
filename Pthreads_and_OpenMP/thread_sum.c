#include <stdio.h>
#include <pthread.h>

#define NUM_THREADS 4
#define ARRAY_SIZE 8

int array[ARRAY_SIZE] = {10, 20, 30, 40, 50, 60, 70, 80};
int partial_sum[NUM_THREADS];

void *calculate_sum(void *arg)
{
    int thread_id = *(int *)arg;
    int start = thread_id * (ARRAY_SIZE / NUM_THREADS);
    int end = start + (ARRAY_SIZE / NUM_THREADS);

    partial_sum[thread_id] = 0;

    for (int i = start; i < end; i++)
    {
        partial_sum[thread_id] += array[i];
    }

    printf("Thread %d calculated sum = %d\n",
           thread_id + 1,
           partial_sum[thread_id]);

    return NULL;
}

int main()
{
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++)
    {
        thread_ids[i] = i;
        pthread_create(
            &threads[i],
            NULL,
            calculate_sum,
            &thread_ids[i]
        );
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    int total_sum = 0;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        total_sum += partial_sum[i];
    }

    printf("Total sum = %d\n", total_sum);
    return 0;
}
