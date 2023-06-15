#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>





__device__ int mandel(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i)
    {

        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

__global__ void mandelKernel(int* img, float x0, float y0, float dx, float dy,int width,int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float x = x0 + i * dx;
    float y = y0 + j * dy;

    int index = (j * width + i);
    img[index] = mandel(x, y, maxIterations);
}

// Host front-end function that allocates the memory and launches the GPU kernel
// 
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    //printf("Start GPU Runtime!!!\n");
    int TILE_Width = 8;
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    int *imggpu,*imghost;
    imghost = (int*)malloc(resX * resY * sizeof(int));
    cudaMalloc(&imggpu, resX * resY * sizeof(int));
    cudaMemcpy(imggpu, imghost, resX * resY * sizeof(int), cudaMemcpyHostToDevice);
    dim3 dimGrid(resX/TILE_Width, resY/TILE_Width);
    dim3 dimBlock(TILE_Width, TILE_Width);
    mandelKernel<<<dimGrid,dimBlock>>>(imggpu, lowerX, lowerY, stepX, stepY,resX, maxIterations);
    cudaMemcpy(imghost, imggpu, resX * resY * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(img, imghost, resX * resY * sizeof(int), cudaMemcpyHostToHost);
    cudaFree(imggpu);
    free(imghost);
}
