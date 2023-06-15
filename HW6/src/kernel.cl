__kernel void convolution(
__global float* inputImage,  __global float* outputImage, __global float* filter, int imageHeight, int imageWidth, int filterWidth){

    int halffilter_size = filterWidth / 2;
    int row = get_global_id(1);
    int col = get_global_id(0);
    float sum = 0;
    for (int k = -halffilter_size; k <= halffilter_size; k++){
        for (int l = -halffilter_size; l <= halffilter_size; l++){
            if (row + k >= 0 && row + k < imageHeight && col + l >= 0 && col + l < imageWidth)
                {
                sum += inputImage[(row + k) * imageWidth + col + l] * filter[(k + halffilter_size) * filterWidth + l + halffilter_size];
            }
        }
    }
    outputImage[row * imageWidth + col] = sum;
}
