__kernel void convolution(
__global float* inputImage,  __global float* outputImage, __global char* filter, int imageHeight, int imageWidth, int filterWidth){

    int halffilter_size = filterWidth / 2;
    int row = get_global_id(1);
    int col = get_global_id(0);
    int k_start = -halffilter_size + row >= 0 ? -halffilter_size : 0;
    int k_end = halffilter_size + row < imageHeight ? halffilter_size : halffilter_size + row - imageHeight - 1;
    int l_start = -halffilter_size + col >= 0 ? -halffilter_size : 0;
    int l_end = halffilter_size + col < imageWidth ? halffilter_size : halffilter_size + col - imageWidth - 1;
    float sum = 0;
    for (int k = k_start; k <= k_end; ++k){
        for (int l = l_start; l <= l_end; ++l) {
                sum += inputImage[(row + k) * imageWidth + col + l] * filter[(k + halffilter_size) * filterWidth + l + halffilter_size];
        }
    }
    outputImage[row * imageWidth + col] = sum;
}
