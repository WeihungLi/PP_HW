#include <stdio.h>
#include <cstdlib>
int  main(int argc, char** argv)
{
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
      fscanf(fp, "%d", &a[i]);/*每次讀取一個數，fscanf函式遇到空格或者換行結束*/
    }

    int** ma = (int**)malloc(a[0] * sizeof(int*));
    for (i = 0; i < a[0]; i++)
    {
        ma[i] = (int*)malloc(a[1] * sizeof(int));
        for (j = 0; j < a[1]; j++)
        {
            fscanf(fp, "%d", &ma[i][j]);/*每次讀取一個數，fscanf函式遇到空格或者換行結束*/
            printf("%d ", ma[i][j]);
        }
        printf("\n");
        fscanf(fp, "\n");
    }

    int** mb = (int**)malloc(a[1] * sizeof(int*));
    for (i = 0; i < a[1]; i++)
    {
        mb[i] = (int*)malloc(a[2] * sizeof(int));
        for (j = 0; j < a[2]; j++)
        {
            fscanf(fp, "%d", &mb[i][j]);/*每次讀取一個數，fscanf函式遇到空格或者換行結束*/
            printf("%d ", mb[i][j]);
        }
        printf("\n");
        fscanf(fp, "\n");
    }
    fclose(fp);

    return 0;
}