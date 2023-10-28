#include <assert.h>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

/**
 * @brief GPU每个线程内部执行的函数, 矩阵中每个对应的数加1
 *
 * @param pMatrx 矩阵
 * @param n 矩阵大小
 * @return __global__ void
 */
__global__ void AddSelf(float *pMatrx, uint32_t n) {
  pMatrx[threadIdx.x + threadIdx.y * n] += 1;
}

/**
 * @brief 打印矩阵
 *
 * @param pMatrx 矩阵
 * @param n 矩阵大小
 */
void PrintMatrix(float *pMatrx, uint32_t n) {
  for (auto i = 0; i < n; ++i) {
    for (auto j = 0; j < n; ++j) {
      std::cout << pMatrx[j + i * n] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

/**
 * @brief 矩阵自增测试
 *
 * @param n 矩阵大小
 */
void MatrixAddSelfTest(uint32_t n) {
  std::cout << std::endl << __func__ << ":" << std::endl;
  // cpu

  // malloc host memory
  float *pHostMatrix = nullptr;
  pHostMatrix = (float *)malloc(sizeof(float) * n * n);
  assert(pHostMatrix);

  // init host memory
  memset(pHostMatrix, 0, sizeof(float) * n * n);
  for (auto i = 0; i < n; ++i) {
    for (auto j = 0; j < n; ++j) {
      pHostMatrix[j + i * n] = 1.0f;
    }
  }
  PrintMatrix(pHostMatrix, n);

  // gpu

  // malloc device memory
  float *pDeviceMatrix = nullptr;
  cudaError_t nCudaErr = cudaSuccess;
  nCudaErr = cudaMalloc(&pDeviceMatrix, sizeof(float) * n * n);
  assert(nCudaErr == cudaSuccess);

  // Copy host data to device memory
  nCudaErr = cudaMemcpy(pDeviceMatrix, pHostMatrix, sizeof(float) * n * n,
                        cudaMemcpyHostToDevice);
  assert(nCudaErr == cudaSuccess);

  // Gpu Execute
  dim3 threadBlocks(1, 1, 1);
  dim3 threads(n, n, 1);
  AddSelf<<<threadBlocks, threads>>>(pDeviceMatrix, n);

  // Copy result to host memory
  nCudaErr = cudaMemcpy(pHostMatrix, pDeviceMatrix, sizeof(float) * n * n,
                        cudaMemcpyDeviceToHost);
  assert(nCudaErr == cudaSuccess);
  // Print
  PrintMatrix(pHostMatrix, n);
  free(pHostMatrix);
  cudaFree(pDeviceMatrix);
}

/**
 * @brief 获取当前Gpu一些属性
 *
 */
void GetGpuDeviceInfoTest() {
  std::cout << std::endl << __func__ << ":" << std::endl;

  cudaDeviceReset();
  int dev = 0;
  cudaDeviceProp devProp{};
  cudaGetDeviceProperties(&devProp, dev);
  std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
  std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
  std::cout << "每个线程块的共享内存大小："
            << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
  std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock
            << std::endl;
  std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor
            << std::endl;
  std::cout << "每个SM的最大线程束数："
            << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
}

int main(int argc, char *argv[]) {

  // 测试
  GetGpuDeviceInfoTest();

  MatrixAddSelfTest(6);

  return 0;
}