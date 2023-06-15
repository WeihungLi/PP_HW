#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
void monteCarlo(int rank, long long int* side_sum, int number_toss);

int fnz (long long int *schedule, long long int *oldschedule, int size)
{
    static double starttime = -1.0;
    int diff = 0;

    for (int i = 0; i < size; i++)
       diff |= (schedule[i] != oldschedule[i]);

    if (diff)
    {
       int res = 0;

       if (starttime < 0.0) starttime = MPI_Wtime();

       //printf("[%6.3f] Schedule:", MPI_Wtime() - starttime);
       for (int i = 0; i < size; i++)
       {
          //printf("\t%lld", schedule[i]);
          if (schedule[i] > 0) {
              res += 1;
          }
          oldschedule[i] = schedule[i];
       }
       //printf("\n");

       return(res == size-1);
    }
    return 0;
}

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Win win;
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    long long int number_in_circle = 0;
    int world_rank, world_size;
    // ---
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // TODO: MPI init

    if (world_rank == 0)
    {
        monteCarlo(world_size, &number_in_circle, tosses / world_size);
        long long int* oldschedule = (long long int*)malloc(world_size * sizeof(long long int));
        // Use MPI to allocate memory for the target window
        long long int *schedule;
        MPI_Alloc_mem(world_size * sizeof(long long int), MPI_INFO_NULL, &schedule);

        for (int i = 0; i < world_size; i++)
        {
            schedule[i] = 0;
            oldschedule[i] = -1;
        }

        // Create a window. Set the displacement unit to sizeof(int) to simplify
        // the addressing at the originator processes
        MPI_Win_create(schedule, world_size * sizeof(long long int), sizeof(long long int), MPI_INFO_NULL,
            MPI_COMM_WORLD, &win);

        int ready = 0;
        while (!ready)
        {
            // Without the lock/unlock schedule stays forever filled with 0s
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
            ready = fnz(schedule, oldschedule, world_size);
            MPI_Win_unlock(0, win);
        }
        //printf("All workers checked in using RMA\n");
        for (int i = 1; i < world_size; i++) {
            number_in_circle += oldschedule[i];
        }
        // Release the window
        MPI_Win_free(&win);
        // Free the allocated memory
        MPI_Free_mem(schedule);
        free(oldschedule);

        //printf("Master done\n");

    }
    else
    {
        long long int one;

        // Worker processes do not expose memory in the window
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        monteCarlo(world_rank, &one, tosses / world_size);
        // Register with the master
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Put(&one, 1, MPI_INT, 0, world_rank, 1, MPI_INT, win);
        MPI_Win_unlock(0, win);

        //printf("Worker %d finished RMA\n", world_rank);

        // Release the window
        MPI_Win_free(&win);

        //printf("Worker %d done\n", world_rank);
    // Workers
    }

    if (world_rank == 0)
    {
        // TODO: handle PI result
        double pi_result = 4 * number_in_circle / (double)tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}

void monteCarlo(int rank,long long int* side_sum,int number_toss) {
    float max = 1.0;
    float min = -1.0;
    long long int sample = 0;
    unsigned seed = rank*rand();
    for (int toss = 0; toss < number_toss; toss++) {
        double x = (max - min) * rand_r(&seed) / (RAND_MAX + 1.0) + min;
        double y = (max - min) * rand_r(&seed) / (RAND_MAX + 1.0) + min;
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1.0)
            sample++;
    }
    *side_sum = sample;

}


