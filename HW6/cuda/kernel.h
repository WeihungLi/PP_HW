#ifndef __HOSTFE__
#define __HOSTFE__
#include <CL/cl.h>

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage);

#endif