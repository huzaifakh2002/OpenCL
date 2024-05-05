#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>

// OpenCL kernel code to convert RGB/BGR to grayscale
const char *kernelCode = 
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

void checkError(cl_int err, const char* message) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: %s (error code: %d)\n", message, err);
        exit(1);
    }
}

int main() {
    cl_int err;

    // Load the image
    cv::Mat colorImage = cv::imread("image.jpg");
    if (colorImage.empty()) {
        fprintf(stderr, "Error: Could not load image.\n");
        return -1;
    }

    int width = colorImage.cols;
    int height = colorImage.rows;
    int channels = colorImage.channels();

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
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "Creating command queue");

    // Compile the OpenCL program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelCode, NULL, &err);
    checkError(err, "Creating program");

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Print the build log in case of failure
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);
        checkError(err, "Building program");
    }

    // Create buffers
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, colorImage.total() * colorImage.elemSize(), NULL, &err);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, colorImage.total() / channels, NULL, &err);
    checkError(err, "Creating buffers");

    // Copy the image data to the input buffer
    err = clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, colorImage.total() * colorImage.elemSize(), colorImage.data, 0, NULL, NULL);
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
    checkError(err, "Enqueueing NDRange kernel");

    // Read the output data
    cv::Mat grayImage(height, width, CV_8UC1);
    err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, grayImage.total(), grayImage.data, 0, NULL, NULL);
    checkError(err, "Reading from output buffer");

    // Save the grayscale image
    cv::imwrite("gray_image.jpg", grayImage);

    // Clean up
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    printf("Grayscale image has been generated.\n");

    return 0;
}

