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
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // TODO: MPI init
    int send;
    if (world_rank == 0){
        monteCarlo(world_size, &send, tosses / world_size);
    }
    else{
        monteCarlo(world_rank, &send, tosses/ world_size);
    }
    // TODO: use MPI_Reduce
    MPI_Reduce(send,&number_in_circle,world_size,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    
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