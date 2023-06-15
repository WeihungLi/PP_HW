#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

__global__ static void convolution(float *inputImage, float *outputImage, char *filter, int filter_width, int image_width, int image_height) {
    int halffilter_size = filter_width / 2;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    int k_start = -halffilter_size + row >= 0 ? -halffilter_size : 0;
    int k_end = halffilter_size + row < image_height ? halffilter_size : halffilter_size + row - image_height - 1;
    int l_start = -halffilter_size + col >= 0 ? -halffilter_size : 0;
    int l_end = halffilter_size + col < image_width ? halffilter_size : halffilter_size + col - image_width - 1;
    for (int k = k_start; k <= k_end; ++k)
        for (int l = l_start; l <= l_end; ++l) 
            sum += inputImage[(row + k) * image_width + col + l] * filter[(k + halffilter_size) * filter_width + l + halffilter_size];

    outputImage[row * image_width + col] = sum;
}

void check_filter(float *filter, char *char_filter, int *filter_width) {
    int check_row = 0;
    int new_filter_width = *filter_width;
    int check_start = 0;
    int check_end = *filter_width - 1;
    bool check = true;
    while(check && check_start < check_end) {
        for (int i = 0; i < *filter_width && check; i++) if(filter[check_start * *filter_width + i] != 0) check = false;  // upper
        for (int i = 0; i < *filter_width && check; i++) if(filter[check_end * *filter_width + i] != 0) check = false;  // lower
        for (int i = 0; i < *filter_width && check; i++) if(filter[i * *filter_width + check_start] != 0) check = false;  // left
        for (int i = 0; i < *filter_width && check; i++) if(filter[i * *filter_width + check_end] != 0) check = false;  // right
        if (check) new_filter_width -= 2;
        check_start++;
        check_end--;
    }
    int char_filter_start = (*filter_width - new_filter_width) % 2 == 0 ? (*filter_width - new_filter_width) / 2 : 0;
    for (register int i = 0; i < new_filter_width; ++i)
        for (register int j = 0; j < new_filter_width; ++j)
            char_filter[i * new_filter_width + j] = filter[((char_filter_start + i) * *filter_width) + char_filter_start + j];

    *filter_width = new_filter_width;
    return;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (int filterWidth, float * filter, int imageHeight, int imageWidth, float * inputImage, float * outputImage) {
    int image_size = imageHeight * imageWidth;
    char *char_filter = (char *)malloc(filterWidth * filterWidth * sizeof(char));

    char *device_filter;
    float *device_input;
    float *device_output;
    cudaMalloc(&device_filter, filterWidth * filterWidth * sizeof(char));
    cudaMalloc(&device_input, image_size * sizeof(float));
    cudaMalloc(&device_output, image_size * sizeof(float));
    cudaMemcpy(device_filter, char_filter, filterWidth * filterWidth * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(device_input, inputImage, image_size * sizeof(float), cudaMemcpyHostToDevice);

    int x_blocks = imageWidth / BLOCK_SIZE;
    int y_blocks = imageHeight / BLOCK_SIZE;
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 num_block(x_blocks, y_blocks);
    convolution<<<num_block, block_size>>>(device_input, device_output, device_filter, filterWidth, imageWidth, imageHeight);
    cudaMemcpy(outputImage, device_output, image_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_filter);
    cudaFree(device_input);
    cudaFree(device_output);

    free(char_filter);
}