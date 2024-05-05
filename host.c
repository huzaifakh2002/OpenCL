#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void checkError(cl_int err, const char* message) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: %s (Error code: %d)\n", message, err);
        exit(1);
    }
}

#pragma pack(push, 1)
typedef struct {
    unsigned char  bfType[2];
    unsigned int   bfSize;
    unsigned short bfReserved1;
    unsigned short bfReserved2;
    unsigned int   bfOffBits;
} BMPHeader;

typedef struct {
    unsigned int   biSize;
    int            biWidth;
    int            biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    unsigned int   biCompression;
    unsigned int   biSizeImage;
    int            biXPelsPerMeter;
    int            biYPelsPerMeter;
    unsigned int   biClrUsed;
    unsigned int   biClrImportant;
} BMPInfoHeader;
#pragma pack(pop)

void loadBMP(const char* filename, unsigned char** imageData, BMPInfoHeader* bmpInfoHeader) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Unable to open file %s.\n", filename);
        exit(1);
    }

    BMPHeader bmpHeader;
    fread(&bmpHeader, sizeof(BMPHeader), 1, file);
    fread(bmpInfoHeader, sizeof(BMPInfoHeader), 1, file);

    if (bmpHeader.bfType[0] != 'B' || bmpHeader.bfType[1] != 'M') {
        fprintf(stderr, "Error: Not a BMP file.\n");
        exit(1);
    }

    size_t dataSize = bmpInfoHeader->biSizeImage;
    *imageData = (unsigned char*)malloc(dataSize);
    fseek(file, bmpHeader.bfOffBits, SEEK_SET);
    fread(*imageData, dataSize, 1, file);

    fclose(file);
}

void saveBMP(const char* filename, const unsigned char* imageData, const BMPInfoHeader* bmpInfoHeader) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Unable to open file %s for writing.\n", filename);
        exit(1);
    }

    BMPHeader bmpHeader = {
        {'B', 'M'},
        (unsigned int)(sizeof(BMPHeader) + sizeof(BMPInfoHeader) + bmpInfoHeader->biSizeImage),
        0,
        0,
        (unsigned int)(sizeof(BMPHeader) + sizeof(BMPInfoHeader))
    };

    fwrite(&bmpHeader, sizeof(BMPHeader), 1, file);
    fwrite(bmpInfoHeader, sizeof(BMPInfoHeader), 1, file);
    fwrite(imageData, bmpInfoHeader->biSizeImage, 1, file);

    fclose(file);
}

const char* kernelCode = 
"__kernel void rgb_to_gray(__global const uchar *input, __global uchar *output, int width, int height, int channels) {\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    int idx = (y * width + x) * channels;\n"
"    uchar blue = input[idx];\n"
"    uchar green = input[idx + 1];\n"
"    uchar red = input[idx + 2];\n"
"    uchar gray = (uchar)(0.299f * red + 0.587f * green + 0.114f * blue);\n"
"    int gray_idx = y * width + x;\n"
"    output[gray_idx] = gray;\n"
"}\n";

int main() {
    cl_int err;

    unsigned char* imageData;
    BMPInfoHeader bmpInfoHeader;
    loadBMP("image.bmp", &imageData, &bmpInfoHeader);

    int width = bmpInfoHeader.biWidth;
    int height = bmpInfoHeader.biHeight;
    int channels = bmpInfoHeader.biBitCount / 8;

    // Get OpenCL platforms
    cl_platform_id platform;
    cl_uint numPlatforms;
    err = clGetPlatformIDs(1, &platform, &numPlatforms);
    checkError(err, "Getting platform");

    // Get a device and create a context
    cl_device_id device;
    cl_uint numDevices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices);
    checkError(err, "Getting device");

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err, "Creating context");

    // Create a command queue
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    checkError(err, "Creating command queue");

    // Compile the OpenCL program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelCode, NULL, &err);
    checkError(err, "Creating program");

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);
        checkError(err, "Building program");
    }

    // Create buffers
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bmpInfoHeader.biSizeImage, NULL, &err);
    checkError(err, "Creating input buffer");

    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height, NULL, &err);
    checkError(err, "Creating output buffer");

    // Copy image data to the input buffer
    err = clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, bmpInfoHeader.biSizeImage, imageData, 0, NULL, NULL);
    checkError(err, "Writing to input buffer");

    // Create the kernel and set its arguments
    cl_kernel kernel = clCreateKernel(program, "rgb_to_gray", &err);
    checkError(err, "Creating kernel");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &width);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &height);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &channels);
    checkError(err, "Setting kernel arguments");

    // Define the global work size
    size_t globalWorkSize[2] = { (size_t)width, (size_t)height };

    // Execute the kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    checkError(err, "Executing kernel");

    unsigned char* grayImage = (unsigned char*)malloc(width * height);
    err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, width * height, grayImage, 0, NULL, NULL);
    checkError(err, "Reading from output buffer");

    saveBMP("gray_image.bmp", grayImage, &bmpInfoHeader);

    // Cleanup
    free(imageData);
    free(grayImage);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    printf("Grayscale image has been generated.\n");

    return 0;
}
