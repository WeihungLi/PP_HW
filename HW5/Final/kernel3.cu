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

__global__ void mandelKernel(int* img, size_t pitch_a, float x0, float y0, float dx, float dy, int width, int heigh, int count, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j % count == 0) {
        int* row_a;
        for (int n = j; n < j + count && n < heigh; n++) {
            float x = x0 + i * dx;
            float y = y0 + n * dy;

            row_a = (int*)((char*)img + n * pitch_a);
            row_a[i] = mandel(x, y, maxIterations);
        }
    }
    else {
        return;
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
// 
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    //int TILE_Width = 16;
    size_t pitch_a;
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    int* imggpu, * imghost;
    cudaHostAlloc((void**)&imghost, resX * resY * sizeof(int), cudaHostAllocMapped);
    cudaMallocPitch((void**)&imggpu, &pitch_a, resX * sizeof(int), resY);
    dim3 dimBlock(16, 16);
    dim3 dimGrid((pitch_a / sizeof(int) + dimBlock.x - 1) / dimBlock.x, (resY + dimBlock.y - 1) / dimBlock.y);
    mandelKernel << <dimGrid, dimBlock >> > (imggpu, pitch_a, lowerX, lowerY, stepX, stepY, resX, resY, 2, maxIterations);
    cudaMemcpy2D(imghost, resX * sizeof(int), imggpu, pitch_a, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(img, resX * sizeof(int), imghost, resX * sizeof(int), resX * sizeof(int), resY, cudaMemcpyHostToHost);
    cudaFree(imggpu);
    cudaFree(imghost);
}
