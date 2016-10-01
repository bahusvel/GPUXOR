//------------------------------------------------------------------------------
//
// Name:       vadd.c
//
// Purpose:    Elementwise addition of two vectors (c = a + b)
//
// HISTORY:    Written by Tim Mattson, December 2009
//             Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated by Tom Deakin, July 2013
//             Updated by Tom Deakin, October 2014
//
//------------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"
#include "read_kernel.h"

// pick up device type from compiler command line or from
// the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

extern double wtime(); // returns time since some fixed past point (wtime.c)
extern int output_device_info(cl_device_id);

//------------------------------------------------------------------------------

#define LENGTH (1024) // length of vectors a, b, and c

cl_device_id pick_device() {
	int i = 0;
	int err; // error code returned from OpenCL calls

	cl_device_id device_id; // compute device id
	// Set up platform and GPU device

	cl_uint numPlatforms;

	// Find number of platforms
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkError(err, "Finding platforms");
	if (numPlatforms == 0) {
		printf("Found 0 platforms!\n");
		exit(-1);
	}

	printf("Found %d platforms\n", numPlatforms);

	// Get all platforms
	cl_platform_id Platform[numPlatforms];
	err = clGetPlatformIDs(numPlatforms, Platform, NULL);
	checkError(err, "Getting platforms");

	// Secure a GPU
	for (i = 0; i < numPlatforms; i++) {
		err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
		if (err == CL_SUCCESS) {
			break;
		}
	}

	if (device_id == NULL)
		checkError(err, "Finding a device");

	err = output_device_info(device_id);
	checkError(err, "Printing device output");
	return device_id;
}

int main(int argc, char **argv) {
	int err; // error code returned from OpenCL calls

	size_t global; // global domain size

	cl_device_id device_id;	// compute device id
	cl_context context;		   // compute context
	cl_command_queue commands; // compute command queue
	cl_program program;		   // compute program
	cl_kernel ko_vadd;		   // compute kernel

	device_id = pick_device();

	// Create a compute context
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	checkError(err, "Creating context");

	// Create a command queue
	commands = clCreateCommandQueue(context, device_id, 0, &err);
	checkError(err, "Creating command queue");

	char *kernel = read_kernel("xor_kernel.cl", NULL);
	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)&kernel,
										NULL, &err);
	checkError(err, "Creating program");

	// Build the program
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t len;
		char buffer[2048];

		printf("Error: Failed to build program executable!\n%s\n",
			   err_code(err));
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
							  sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		return EXIT_FAILURE;
	}

	// Create the compute kernel from the program
	ko_vadd = clCreateKernel(program, "vadd", &err);
	checkError(err, "Creating kernel");

	cl_mem data_buf; // device memory used for the input  a vector
	cl_mem key_buf;  // device memory used for the input  b vector
	// Create the input (a, b) and output (c) arrays in device memory
	size_t data_length = 0;
	char *data = read_kernel("add_kernel.cl", &data_length);
	char *key = "hello";
	size_t key_length = strlen(key);

	data_buf =
		clCreateBuffer(context, CL_MEM_READ_WRITE, data_length, NULL, &err);
	checkError(err, "Creating buffer data_buf");

	key_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, key_length, NULL, &err);
	checkError(err, "Creating buffer key_buf");

	// Write a and b vectors into compute device memory
	err = clEnqueueWriteBuffer(commands, data_buf, CL_TRUE, 0, data_length,
							   data, 0, NULL, NULL);
	checkError(err, "Copying h_a to device at d_a");
	printf("Keylen %lu\n", key_length);
	err = clEnqueueWriteBuffer(commands, key_buf, CL_TRUE, 0, key_length, key,
							   0, NULL, NULL);
	checkError(err, "Copying h_b to device at d_b");

	// Set the arguments to our compute kernel
	err = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &data_buf);
	err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &key_buf);
	err |= clSetKernelArg(ko_vadd, 2, sizeof(unsigned int), &data_length);
	err |= clSetKernelArg(ko_vadd, 3, sizeof(unsigned int), &key_length);
	checkError(err, "Setting kernel arguments");

	double rtime = wtime();

	// Execute the kernel over the entire range of our 1d input data set
	// letting the OpenCL runtime choose the work-group size
	global = data_length;
	err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0,
								 NULL, NULL);
	checkError(err, "Enqueueing kernel");

	// Wait for the commands to complete before stopping the timer
	err = clFinish(commands);
	checkError(err, "Waiting for kernel to finish");

	rtime = wtime() - rtime;
	printf("\nThe kernel ran in %lf seconds\n", rtime);

	// Read back the results from the compute device
	err = clEnqueueReadBuffer(commands, data_buf, CL_TRUE, 0, data_length, data,
							  0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to read output array!\n%s\n", err_code(err));
		exit(1);
	}

	// cleanup then shutdown
	clReleaseMemObject(data_buf);
	clReleaseMemObject(key_buf);
	clReleaseProgram(program);
	clReleaseKernel(ko_vadd);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	return 0;
}
