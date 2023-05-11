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
    int a[3];
    int i, j;
    FILE* fp = fopen("/home/.grade/HW4/data-set/data1_1", "r"); //開啟檔案
    if (fp == NULL)
    {
        printf("no file!!\n");
        return -1;
    }
    for (i = 0; i < 3; i++)
    {
        if (i == 0) {
            fscanf(fp, "%d", n_ptr);/*每次讀取一個數，fscanf函式遇到空格或者換行結束*/
        }
        else if (i == 1) {
            fscanf(fp, "%d", m_ptr);/*每次讀取一個數，fscanf函式遇到空格或者換行結束*/
        }
        else {
            fscanf(fp, "%d", l_ptr);/*每次讀取一個數，fscanf函式遇到空格或者換行結束*/
        }
    }

    int* ma = (int**)malloc(a[0]* a[1] * sizeof(int*));
    for (i = 0; i < a[0] * a[1]; i++)
    {
            fscanf(fp, "%d", &ma[i]);/*每次讀取一個數，fscanf函式遇到空格或者換行結束*/
    }

    int* mb = (int**)malloc(a[1]* a[2] * sizeof(int*));
    for (i = 0; i < a[1] * a[2]; i++)
    {
            fscanf(fp, "%d", &mb[i]);/*每次讀取一個數，fscanf函式遇到空格或者換行結束*/
    }
    fclose(fp);
    **a_mat_ptr = ma;
    **b_mat_ptr = mb;
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

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int* C = (int*)calloc((n * l),sizeof(int));

    int local_n = N / size;
    int* local_A = (int*)malloc(local_n * m * sizeof(int));
    MPI_Scatter(a_mat, local_n * m, MPI_INT, local_A, local_n * m, MPI_INT, 0, MPI_COMM_WORLD);
    // Broadcast matrix B to all processes
    MPI_Bcast(b_mat, m * l, MPI_INT, 0, MPI_COMM_WORLD);


    for (i = 0; i < local_n; i++) {
        for (j = 0; j < l; j++) {
            for (k = 0; k < m; k++) {
                C[i * l + j] += local_A[i * m + k] * b_mat[k * l + j];
            }
        }
    }

    MPI_Gather(C, local_n * l, MPI_INT, C, local_n * l, MPI_INT, 0, MPI_COMM_WORLD);


    free(C);
    free(local_A);
}

// Remember to release your allocated memory
void destruct_matrices(int* a_mat, int* b_mat) {
    free(a_mat);
    free(b_mat);
}





