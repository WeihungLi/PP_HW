#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

#define WGX 4
#define WGY 4

unsigned int roundUp(unsigned int value, unsigned int multiple) {

    // Determine how far past the nearest multiple the value is
    unsigned int remainder = value % multiple;

    // Add the difference to make the value a multiple
    if (remainder != 0) {
        value += (multiple - remainder);
    }

    return value;
}

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{	

    // Size of the input and output images on the host
    int dataSize = imageHeight * imageWidth * sizeof(float);

    // Pad the number of columns 
    int deviceWidth = imageWidth;
    int deviceHeight = imageHeight;
    // Size of the input and output images on the device
    int deviceDataSize = imageHeight * deviceWidth * sizeof(float);

    int paddingPixels = (int)(filterWidth / 2) * 2;
    // Set up the OpenCL environment

    cl_event timing_event;
    cl_command_queue queue;
    queue = clCreateCommandQueue(*context, *device, CL_QUEUE_PROFILING_ENABLE, NULL);
    // Create memory buffers
    cl_mem d_inputImage;
    cl_mem d_outputImage;
    cl_mem d_filter;
    d_inputImage = clCreateBuffer(*context, CL_MEM_READ_ONLY,
        deviceDataSize, NULL, NULL);
    d_outputImage = clCreateBuffer(*context, CL_MEM_WRITE_ONLY,
        deviceDataSize, NULL, NULL);
    d_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY,
        filterWidth*filterWidth * sizeof(float), NULL, NULL);

    // Write input data to the device
    clEnqueueWriteBuffer(queue, d_inputImage, CL_TRUE, 0, deviceDataSize,
        inputImage, 0, NULL, NULL);
    // Write the filter to the device
    clEnqueueWriteBuffer(queue, d_filter, CL_TRUE, 0,
        filterWidth*filterWidth * sizeof(float), filter, 0, NULL, NULL);
    // Create and compile the program
    // Create the kernel
    cl_kernel kernel;
    // Only the host-side code differs for the aligned reads
    kernel = clCreateKernel(*program, "convolution", NULL);

    // Selected work group size is 16x16
    int wgWidth = WGX;
    int wgHeight = WGY;

    // When computing the total number of work items, the 
    // padding work items do not need to be considered
    // Size of a work group
    size_t localSize[2] = { wgWidth, wgHeight };
    // Size of the NDRange
    size_t globalSize[2] = { imageWidth, imageHeight};

    // Set the kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_inputImage);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_outputImage);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_filter);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &deviceHeight);
    clSetKernelArg(kernel, 4, sizeof(cl_int), &deviceWidth);
    clSetKernelArg(kernel, 5, sizeof(cl_int), &filterWidth);
    // Execute the kernel
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize,
        NULL, 0, NULL, &timing_event);
    // Wait for kernel to complete
    //clFinish(queue);

    // Read back the output image
    clEnqueueReadBuffer(queue, d_outputImage, CL_TRUE, 0,
        deviceDataSize, outputImage, 0, NULL, NULL);
    

    // Free OpenCL objects
    clReleaseMemObject(d_inputImage);
    clReleaseMemObject(d_outputImage);
    clReleaseMemObject(d_filter);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);

    return 0;

}