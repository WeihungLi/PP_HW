#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
void monteCarlo(int rank, long long int* side_sum, int number_toss);
int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    long long int number_in_circle = 0;
    int world_rank, world_size;
    int source;
    // ---
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // TODO: MPI init

    if (world_rank > 0)
    {
        //printf("The rank %d launch \n", world_rank);
        MPI_Request request;
        long long int side_sum = 0;
        monteCarlo(world_rank, &side_sum, tosses/ world_size);
        printf("The rank %d send the sum of toss %lld\n", world_rank, side_sum);
        MPI_Isend(&side_sum, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD, &request);
        // TODO: handle workers
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
    else if (world_rank == 0)
    {
        long long int* sample = (long long int*)malloc((world_size - 1) * sizeof(long long int));
        monteCarlo(world_size, &number_in_circle, tosses / world_size);
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        MPI_Request* requests = (MPI_Request*)malloc((world_size-1)*sizeof(MPI_Request));
        for (source = 1; source < world_size; source++) {
            MPI_Irecv(&sample[source-1], 1, MPI_LONG, source, 0, MPI_COMM_WORLD,
                &requests[source-1]);
        }
        MPI_Waitall(world_size-1,requests, MPI_STATUS_IGNORE);

        for (int i = 0; i < world_size-1; i++) {
            number_in_circle += sample[i];
        }
    }

    if (world_rank == 0)
    {
        // TODO: PI result
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