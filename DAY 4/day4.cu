#include <stdio.h>
#include <cuda_runtime.h>

__global__ void partialSumKernel(int *input, int *output, int n) {
    extern __shared__ int sharedMemory[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    if (index < n)
        sharedMemory[tid] = input[index];
    else
        sharedMemory[tid] = 0;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int temp = 0;
        if (tid >= stride) temp = sharedMemory[tid - stride];
        __syncthreads();
        sharedMemory[tid] += temp;
        __syncthreads();
    }

    if (index < n)
        output[index] = sharedMemory[tid];
}

int main() {
    const int N = 16;
    const int blockSize = 8;

    int h_input[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int h_output[N] = {0};

    int *d_input, *d_output;
    size_t size = N * sizeof(int);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    partialSumKernel<<<(N + blockSize - 1) / blockSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    printf("Input: ");
    for (int i = 0; i < N; i++) printf("%d ", h_input[i]);
    printf("\nOutput: ");
    for (int i = 0; i < N; i++) printf("%d ", h_output[i]);
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
