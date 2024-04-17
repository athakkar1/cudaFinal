
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define BUFFER_SIZE 16384 * 4
#define MASK_LEN 59
#define OUTPUT_SIZE 966
#define B_SIZE OUTPUT_SIZE + (MASK_LEN - 1)
#define BLOCK_SIZE 512
#define FFT_SIZE 16384 * 4
int *buffer, *buffer_out;
__global__ void convolutionKernel(int* input, int* output);
__global__ void convolutionKernelOptimized(int* input, int* output);
__global__ void DFT_kernel(int* input, cuFloatComplex* output);
__global__ void IDFT_kernel(cuFloatComplex* input, int* output);
__constant__ double mask[MASK_LEN];
void cleanup() {
  fprintf(stderr, "Cleaning up...\n");
  checkCudaErrors(cudaFreeHost(buffer));
  checkCudaErrors(cudaFreeHost(buffer_out));
}
void handle_signal(int signal) {
  // Perform cleanup
  cleanup();

  // Exit the program
  exit(1);
}
void clear_stdin() {
  int c;
  while ((c = getchar()) != '\n' && c != EOF) {
  }
}
void convolution() {
  int *input_d, *output_d;
  checkCudaErrors(cudaMalloc(&input_d, BUFFER_SIZE * sizeof(int)));
  checkCudaErrors(cudaMemcpy(input_d, buffer, BUFFER_SIZE * sizeof(int),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&output_d, BUFFER_SIZE * sizeof(int)));
  // uint32_t BLOCK_SIZE = OUTPUT_SIZE + MASK_LEN - 1;
  dim3 dimBlock(B_SIZE);
  dim3 dimGrid((BUFFER_SIZE + OUTPUT_SIZE - 1) / OUTPUT_SIZE);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);
  convolutionKernelOptimized<<<dimGrid, dimBlock>>>(input_d, output_d);
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  fprintf(stderr, "Milliseconds for Kernel: %f", milliseconds);
  checkCudaErrors(cudaMemcpy(buffer_out, output_d, BUFFER_SIZE * sizeof(int),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(input_d));
  checkCudaErrors(cudaFree(output_d));
}
void dft() {
  cuFloatComplex* dftinput_d;
  int *input_d, *output_d;
  checkCudaErrors(cudaMalloc(&input_d, BUFFER_SIZE * sizeof(int)));
  checkCudaErrors(cudaMemcpy(input_d, buffer, BUFFER_SIZE * sizeof(int),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&output_d, BUFFER_SIZE * sizeof(int)));
  checkCudaErrors(cudaMalloc(&dftinput_d, FFT_SIZE * sizeof(cuFloatComplex)));
  dim3 dimBlock_input(BLOCK_SIZE);
  dim3 dimGrid(FFT_SIZE / BLOCK_SIZE);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);
  DFT_kernel<<<dimGrid, dimBlock_input>>>(input_d, dftinput_d);
  dim3 dimGridInverse(BUFFER_SIZE / BLOCK_SIZE);
  IDFT_kernel<<<dimGridInverse, dimBlock_input>>>(dftinput_d, output_d);
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  fprintf(stderr, "Milliseconds for Kernel: %f", milliseconds);
  checkCudaErrors(cudaMemcpy(buffer_out, output_d, BUFFER_SIZE * sizeof(int),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(dftinput_d));
  checkCudaErrors(cudaFree(input_d));
  checkCudaErrors(cudaFree(output_d));
}

__global__ void convolutionKernel(int* input, int* output) {
  int output_index = blockIdx.x;
  int mask_index = threadIdx.x;
  int buffer_index = (threadIdx.x - (MASK_LEN / 2)) + output_index;
  int mult = 0;
  if (buffer_index > -1 && buffer_index < BUFFER_SIZE) {
    mult = int(input[buffer_index] * mask[mask_index]);
    atomicAdd(&output[output_index], mult);
  }
  if ((threadIdx.x - MASK_LEN / 2) == 0) {
    output[output_index] = output[output_index] * 5;
  }
}

__global__ void convolutionKernelOptimized(int* input, int* output) {
  __shared__ int input_shared[B_SIZE];
  int output_index = blockIdx.x * OUTPUT_SIZE + threadIdx.x;
  int input_index = output_index - (MASK_LEN / 2);
  float shift_amount = 2.0f * M_PI * 500.0f / 44100;
  cuFloatComplex shift = make_cuFloatComplex(cos(shift_amount * output_index),
                                             sin(shift_amount * output_index));
  if (input_index > -1 && input_index < BUFFER_SIZE) {
    input_shared[threadIdx.x] = input[input_index] * shift.x;
  } else {
    input_shared[threadIdx.x] = 0;
  }
  int sum = 0;
  __syncthreads();
  if (threadIdx.x < OUTPUT_SIZE) {
    for (int i = 0; i < MASK_LEN; i++) {
      sum += mask[i] * input_shared[threadIdx.x + i];
    }
    if (output_index < BUFFER_SIZE) {
      output[output_index] = sum * 4;
    }
  }
  __syncthreads();
}
__global__ void DFT_kernel(int* input, cuFloatComplex* output) {
  __shared__ int input_shared[BLOCK_SIZE];
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  float c = -2 * M_PI * k / BUFFER_SIZE;
  cuFloatComplex sum = make_cuFloatComplex(0, 0);
  for (int p = 0; p < BUFFER_SIZE / BLOCK_SIZE; p++) {
    input_shared[threadIdx.x] = input[p * BLOCK_SIZE + threadIdx.x];
    __syncthreads();
    for (int n = 0; n < BLOCK_SIZE; n++) {
      float imag, real;
      sincosf(c * ((p * BLOCK_SIZE) + n), &imag, &real);
      sum = cuCaddf(sum, make_cuFloatComplex(real * input_shared[n],
                                             imag * input_shared[n]));
    }
    __syncthreads();
  }
  output[k] = sum;
  // output_time[k] =
  // int(sqrt((output[k].x * output[k].x) + (output[k].y * output[k].y)));
#if 0 
if ((k * 44100) / BUFFER_SIZE < 6174) { output[k] = sum; }
  else {
    output[k] = make_cuFloatComplex(0, 0);
  }
#endif
}
__global__ void IDFT_kernel(cuFloatComplex* input, int* output) {
  __shared__ cuFloatComplex input_shared[BLOCK_SIZE];
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  float c = 2 * M_PI * k / FFT_SIZE;
  int sum = 0;
  for (int p = 0; p < FFT_SIZE / BLOCK_SIZE; p++) {
    input_shared[threadIdx.x] = input[p * BLOCK_SIZE + threadIdx.x];
    __syncthreads();
    for (int n = 0; n < BLOCK_SIZE; n++) {
      float imag, real;
      sincosf(c * ((p * BLOCK_SIZE) + n), &imag, &real);
      sum = sum + cuCmulf(input_shared[n], make_cuFloatComplex(real, imag)).x;
    }
    __syncthreads();
  }
  output[k] = sum / FFT_SIZE;
}
int main() {
  signal(SIGINT, handle_signal);
  size_t buffer_size = BUFFER_SIZE * sizeof(int);
  checkCudaErrors(cudaMallocHost((void**)&buffer, buffer_size));
  checkCudaErrors(cudaMallocHost((void**)&buffer_out, buffer_size));
  double mask_host[] = {
      0.000000000000000000,  0.000004438834203373,  0.000049053082052918,
      -0.000015130777575385, -0.000229225590281534, -0.000048279893872748,
      0.000581178040493617,  0.000326679950226058,  -0.001107313250291415,
      -0.001027132488276441, 0.001720767967129273,  0.002409735576127541,
      -0.002184047527103399, -0.004739632421046886, 0.002055147463495010,
      0.008208556378427059,  -0.000653542275366089, -0.012845447832394764,
      -0.002965965598441678, 0.018445807535328835,  0.010104312716340521,
      -0.024549571108526527, -0.022821285652037081, 0.030485959851263816,
      0.045627485282245014,  -0.035483947448544301, -0.094461757802380739,
      0.038825196141212139,  0.314286136325509613,  0.460003649044168850,
      0.314286136325509613,  0.038825196141212139,  -0.094461757802380739,
      -0.035483947448544294, 0.045627485282245007,  0.030485959851263827,
      -0.022821285652037091, -0.024549571108526524, 0.010104312716340521,
      0.018445807535328852,  -0.002965965598441680, -0.012845447832394771,
      -0.000653542275366089, 0.008208556378427068,  0.002055147463495013,
      -0.004739632421046888, -0.002184047527103402, 0.002409735576127543,
      0.001720767967129272,  -0.001027132488276442, -0.001107313250291414,
      0.000326679950226057,  0.000581178040493616,  -0.000048279893872748,
      -0.000229225590281534, -0.000015130777575385, 0.000049053082052918,
      0.000004438834203373,  0.000000000000000000,
  };
  checkCudaErrors(
      cudaMemcpyToSymbol(mask, mask_host, MASK_LEN * sizeof(double)));
  char ready_signal[5];
  while (1) {
    fprintf(stderr, "Waiting for ready signal...\n");
    fgets(ready_signal, sizeof(ready_signal), stdin);
    if (strcmp(ready_signal, "RDY\n") != 0) {
      fprintf(stderr, "Wrong ready_signal: %s\n", ready_signal);
      return 1;
    } else {
      fprintf(stderr, "Right ready_signal: %s\n", ready_signal);
    }
    for (int i = 0; i < BUFFER_SIZE; i++) {
      if (fscanf(stdin, "%d", &buffer[i]) != 1) {
        if (feof(stdin)) {
          clearerr(stdin);
        } else {
          fprintf(stderr, "Error: could not read value %d\n", i);
          return 1;
        }
      }
    }
    convolution();
    printf("RDY\n");
    fflush(stdout);
    for (int i = 0; i < BUFFER_SIZE; i++) {
      fprintf(stdout, "%d\n", buffer_out[i]);
    }
    fflush(stdout);
    memset(buffer, 0, sizeof(buffer));
    memset(ready_signal, 0, sizeof(ready_signal));
    clear_stdin();
  }

  return 0;
}