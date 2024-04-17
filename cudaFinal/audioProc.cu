#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdio.h>
#include <time.h>

#include <ctime>

int main(int argc, char** argv) {
  int dimX, dimY, dimK;
  if (checkCmdLineFlag(argc, (const char**)argv, "dimX")) {
    dimX = getCmdLineArgumentInt(argc, (const char**)argv, "dimX");
  }
  // height of Matrix A
  if (checkCmdLineFlag(argc, (const char**)argv, "dimY")) {
    dimY = getCmdLineArgumentInt(argc, (const char**)argv, "dimY");
  }
  if (checkCmdLineFlag(argc, (const char**)argv, "dimK")) {
    dimK = getCmdLineArgumentInt(argc, (const char**)argv, "dimK");
  }
  printf("%d, %d, %d\n", dimX, dimY, dimK);
  convolution(dimX, dimY, dimK);
  printf("\n");
  return 0;
}