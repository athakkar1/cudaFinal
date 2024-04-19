
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
#define BUFFER_SIZE 65536
#define MASK_LEN 59
#define OUTPUT_SIZE 966
#define B_SIZE OUTPUT_SIZE + (MASK_LEN - 1)
#define BLOCK_SIZE 512
#define FFT_SIZE 32768
int *buffer, *buffer_out;
int millisecondSum = 0;
int counter = 0;
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
  counter++;
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
  // fprintf(stderr, "Milliseconds for Kernel: %f", milliseconds);
  millisecondSum += milliseconds;
  if (counter == 5) {
    millisecondSum = millisecondSum / (5);
    float gflops = (2.0f * BUFFER_SIZE * MASK_LEN) / milliseconds / 1e6;
    fprintf(stderr, "GFLOPS for Convolution Kernel: %f\n", gflops);
    counter = 0;
    millisecondSum = 0;
  }
  checkCudaErrors(cudaMemcpy(buffer_out, output_d, BUFFER_SIZE * sizeof(int),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(input_d));
  checkCudaErrors(cudaFree(output_d));
}
void dft() {
  counter++;
  cuFloatComplex* dftinput_d;
  int *input_d, *output_d;
  // allocate memory for buffer of audio
  checkCudaErrors(cudaMalloc(&input_d, BUFFER_SIZE * sizeof(int)));
  // samples copy buffer to device memory
  checkCudaErrors(cudaMemcpy(input_d, buffer, BUFFER_SIZE * sizeof(int),
                             cudaMemcpyHostToDevice));
  // allocate device memory for output
  checkCudaErrors(cudaMalloc(&output_d, BUFFER_SIZE * sizeof(int)));
  // allocate memory for DFT points
  checkCudaErrors(cudaMalloc(&dftinput_d, FFT_SIZE * sizeof(cuFloatComplex)));
  // initialize block dim3
  dim3 dimBlock_input(BLOCK_SIZE);
  // each thread responsible for computing one dft point, therefore launch
  // enough threads for FFT_SIZE
  dim3 dimGrid((FFT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);
  DFT_kernel<<<dimGrid, dimBlock_input>>>(input_d, dftinput_d);
  dim3 dimGridInverse((BUFFER_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
  IDFT_kernel<<<dimGridInverse, dimBlock_input>>>(dftinput_d, output_d);
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  // fprintf(stderr, "Milliseconds for DFT Kernel: %f", milliseconds);
  millisecondSum += milliseconds;
  if (counter == 5) {
    millisecondSum = millisecondSum / (5);
    float gflops = (4.0f * BUFFER_SIZE * FFT_SIZE) / milliseconds / 1e6;
    fprintf(stderr, "GFLOPS for DFT Kernel: %f\n", gflops);
    counter = 0;
    millisecondSum = 0;
  }
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
  // float shift_amount = 2.0f * M_PI * 500.0f / 44100;
  // cuFloatComplex shift = make_cuFloatComplex(cos(shift_amount *
  // output_index),
  //                                            sin(shift_amount *
  //                                            output_index));
  if (input_index > -1 && input_index < BUFFER_SIZE) {
    input_shared[threadIdx.x] = input[input_index];  // * shift.x;
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
    input_shared[threadIdx.x] = input[(p * BLOCK_SIZE) + threadIdx.x];
    __syncthreads();
    for (int n = 0; n < BLOCK_SIZE; n++) {
      float imag, real;
      sincosf(c * ((p * BLOCK_SIZE) + n), &imag, &real);
      sum = cuCaddf(sum, make_cuFloatComplex(real * input_shared[n],
                                             imag * input_shared[n]));
    }
    __syncthreads();
  }
  // output[k] = sum;D
#if 1
  if ((k * 44100) / FFT_SIZE > 3000) {
    output[k] = sum;
  } else {
    output[k] = make_cuFloatComplex(0, 0);
  }
#endif
}
__global__ void IDFT_kernel(cuFloatComplex* input, int* output) {
  __shared__ cuFloatComplex input_shared[BLOCK_SIZE];
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  float c = 2 * M_PI * k / FFT_SIZE;
  float sum = 0;
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
  output[k] = int(sum / FFT_SIZE);
}
int main() {
  signal(SIGINT, handle_signal);
  size_t buffer_size = BUFFER_SIZE * sizeof(int);
  checkCudaErrors(cudaMallocHost((void**)&buffer, buffer_size));
  checkCudaErrors(cudaMallocHost((void**)&buffer_out, buffer_size));
  double mask_host[] = {
      0.000000000000000000,  -0.000012033708518804, -0.000027135447534361,
      0.000058157163173842,  0.000229216710532700,  0.000185570216202303,
      -0.000321499191608881, -0.000885631478862312, -0.000537999764379554,
      0.001027092699182624,  0.002341847097367069,  0.001151243636007275,
      -0.002598060365520312, -0.005145182314934716, -0.002055067851121594,
      0.005720247141516842,  0.010080909127449795,  0.003200725121983794,
      -0.011538034732601451, -0.018445092980999400, -0.004446637242013353,
      0.022402476943542249,  0.033286906592325048,  0.005581436075138684,
      -0.045625717763144058, -0.066643169632041310, -0.006379031058612831,
      0.131810456024836181,  0.277589245940945584,  0.339989526083377847,
      0.277589245940945639,  0.131810456024836181,  -0.006379031058612831,
      -0.066643169632041310, -0.045625717763144058, 0.005581436075138686,
      0.033286906592325062,  0.022402476943542242,  -0.004446637242013354,
      -0.018445092980999413, -0.011538034732601456, 0.003200725121983796,
      0.010080909127449795,  0.005720247141516848,  -0.002055067851121598,
      -0.005145182314934718, -0.002598060365520316, 0.001151243636007277,
      0.002341847097367068,  0.001027092699182625,  -0.000537999764379553,
      -0.000885631478862312, -0.000321499191608881, 0.000185570216202303,
      0.000229216710532700,  0.000058157163173842,  -0.000027135447534361,
      -0.000012033708518804, 0.000000000000000000,
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
    dft();
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