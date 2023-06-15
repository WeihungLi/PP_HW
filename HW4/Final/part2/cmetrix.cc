#include <mpi.h>
#include <cstdio>


// Read size of matrix_a and matrix_b (n, m, l) and whole data of matrixes from stdin
//
// n_ptr:     pointer to n
// m_ptr:     pointer to m
// l_ptr:     pointer to l
// a_mat_ptr: pointer to matrix a (a should be a continuous memory space for placing n * m elements of int)
// b_mat_ptr: pointer to matrix b (b should be a continuous memory space for placing m * l elements of int)
void construct_matrices(int* n_ptr, int* m_ptr, int* l_ptr,
    int** a_mat_ptr, int** b_mat_ptr) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank==0){
    int i, j;

    for (i = 0; i < 3; i++)
    {
        if (i == 0) {
            scanf("%d", n_ptr);
            //printf("%d\n",*n_ptr);
        }
        else if (i == 1) {
            scanf("%d", m_ptr);
            //printf("%d\n",*m_ptr);
        }
        else {
            scanf("%d", l_ptr);
            //printf("%d\n",*l_ptr);
        }
    }
    //printf("a_mat_ptr construct!!\n");
    (*a_mat_ptr) = (int*)malloc((*n_ptr) * (*m_ptr) * sizeof(int));
    //printf("a_mat_ptr construct!! Size:%ld,len:%d\n",sizeof(*a_mat_ptr), (*n_ptr) * (*m_ptr));
    for (i = 0; i < *n_ptr; i++)
    {
        for (j = 0;j<*m_ptr;j++){
            scanf("%d", &((*a_mat_ptr)[i* (*m_ptr)+j]));
        }
    }
    //printf("b_mat_ptr construct!!\n");
    (*b_mat_ptr) = (int*)malloc((*m_ptr) * (*l_ptr) * sizeof(int));
    for (i = 0; i < *m_ptr; i++)
    {
        for (j = 0;j<*l_ptr;j++){
            scanf("%d", &((*b_mat_ptr)[i* (*l_ptr) +j]));
        }
    }
    //printf("End!!!! a_mat_ptr:%d, a_mat_ptr:%d, a_mat_ptr:%d\n", tempa[0], tempa[1], tempa[2]);
    }

}

// Just matrix multiplication (your should output the result in this function)
// 
// n:     row number of matrix a
// m:     col number of matrix a / row number of matrix b
// l:     col number of matrix b
// a_mat: a continuous memory placing n * m elements of int
// b_mat: a continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l,
    const int* a_mat, const int* b_mat) {

    int rank, size;
    int i, j, k;
    int a = (int)n, b = (int)m, c = (int)l;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Bcast(&a, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    int rest = a % size;
    int local_n = a / size;
    //printf("local_n :%d \n", local_n);
    int* local_A = (int*)malloc((local_n * b)* sizeof(int));
    int* A = (int*)malloc((a * b) * sizeof(int));
    int* B = (int*)malloc((b * c) * sizeof(int));
    int* C = (int*)calloc((a * c) , sizeof(int));
    if (rank == 0) {
        A = (int*)a_mat;
        B = (int*)b_mat;
    }
    MPI_Scatter(A, local_n * b, MPI_INT, local_A, local_n * b, MPI_INT, 0, MPI_COMM_WORLD);
    //printf("Rank: % d, A : % d\n", rank, local_A[0]);
    
    // Broadcast matrix B to all processes
    MPI_Bcast(B, b * c, MPI_INT, 0, MPI_COMM_WORLD);
    //printf("Rank: % d, B : % d\n", rank, B[10]);
    
    for (i = 0; i < local_n; i++) {
        for (k = 0; k < b; k++) {
            for (j = 0; j < c; j++) {
                C[i * c + j] += (local_A[i * b + k] * B[k * c + j]);
            }
        }
    }
    if (rank == 0) {
        for (i = 0; i < rest; i++) {
            for (k = 0; k < b; k++) {
                for (j = 0; j < c; j++) {
                    C[i * c + j + local_n * size * c] += (A[i * b + k + local_n * size * b] * B[k * c + j]);
                }
            }
        }
    }
    MPI_Gather(C, local_n * c, MPI_INT, C, local_n * c, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (i = 0; i < a; i++) {
            for (j = 0; j < c; j++) {
                printf( "%d ", C[i * c + j]);  // Write integer to file
            }
            printf("\n");
        }
    }
    
    free(C);
    free(local_A);
    
}

// Remember to release your allocated memory
void destruct_matrices(int* a_mat, int* b_mat) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0){
        free(a_mat);
        free(b_mat); 
    }
}





