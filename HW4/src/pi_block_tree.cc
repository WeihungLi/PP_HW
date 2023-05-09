#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>

void monteCarlo(int rank, long long int* side_sum, int number_toss);
int* getNumsBinaryReduce(int maxVal);
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

    // TODO: MPI init
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int* numsDict = getNumsBinaryReduce(world_size);
    // TODO: binary tree redunction
    if (world_rank%2 != 0)
    {
        //printf("The %d Launch!!!\n", world_rank);
        long long int side_sum = 0;
        monteCarlo(world_rank, &side_sum, tosses / world_size);
        MPI_BSend(&side_sum, 1, MPI_LONG, world_rank-1, 0, MPI_COMM_WORLD);
    }
    else if (world_rank % 2 == 0 && world_rank != 0) {
        //printf("The %d Launch!!!\n", world_rank);
        long long int side_sum = 0;
        long long int sample;
        int recvTarget = (world_rank + 1);
        //printf("The recvTarget %d,Rank:%d!!!\n", recvTarget, world_rank);
        monteCarlo(world_rank, &side_sum, tosses / world_size);
        MPI_Recv(&sample, 1, MPI_LONG, recvTarget, 0, MPI_COMM_WORLD,&status);
        side_sum += sample;
        for (int nums = 0; nums < numsDict[world_rank]-1; nums++) {
            recvTarget += pow(2, nums);
            //printf("The recvTarget %d,Rank:%d!!!\n", recvTarget, world_rank);
            MPI_Recv(&sample, 1, MPI_LONG, recvTarget, 0, MPI_COMM_WORLD,
                &status);
            side_sum += sample;
        }
        //printf("The Send %d,Rank:%d!!!\n", recvTarget, world_rank);
        MPI_BSend(&side_sum, 1, MPI_LONG, world_rank- pow(2,numsDict[world_rank]), 0, MPI_COMM_WORLD);
    }
    else
    {
        //printf("The %d Launch!!!\n", world_rank);
        long long int sample;
        int recvTarget = (world_rank + 1);
        //printf("The recvTarget %d,Rank:%d!!!\n", recvTarget, world_rank);
        monteCarlo(world_size, &number_in_circle, tosses / world_size);
        MPI_Recv(&sample, 1, MPI_LONG, recvTarget, 0, MPI_COMM_WORLD,&status);
        number_in_circle += sample;
        for (int nums = 0; nums < numsDict[world_rank] - 1; nums++) {
            recvTarget += pow(2, nums);
            //printf("The recvTarget %d,Rank:%d!!!\n", recvTarget, world_rank);
            MPI_Recv(&sample, 1, MPI_LONG, recvTarget, 0, MPI_COMM_WORLD,
                &status);
            number_in_circle += sample;
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


void monteCarlo(int rank, long long int* side_sum, int number_toss) {
    float max = 1.0;
    float min = -1.0;
    long long int sample = 0;
    unsigned seed = rank * rand(); // 取得時間序列
    //printf("seed : %d\n", seed);
    for (int toss = 0; toss < number_toss; toss++) {
        double x = (max - min) * rand_r(&seed) / (RAND_MAX + 1.0) + min;
        double y = (max - min) * rand_r(&seed) / (RAND_MAX + 1.0) + min;
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1.0)
            sample++;
    }
    *side_sum = sample;

}


int* getNumsBinaryReduce(int maxVal) {
    int* nums = (int*)calloc((maxVal),sizeof(int));
    for (int i = 0; i < maxVal-1; i += 2) {
        int Rbound = maxVal-1;
        int Lbound = 0;
        while (Rbound > Lbound) {
            if ((Rbound + Lbound) / 2.0 > i) {
                Rbound = ((Rbound + Lbound) / 2);
                if (Lbound == i) {
                    nums[i]++;
                }
            }
            else {
                Lbound = ((Rbound + Lbound) / 2) + 1;
            }
        }
    }
    return nums;
}

