
#include <iostream>
#include <math.h>

// Macro for checking CUDA errors
#define CUDA_CHECK(call) \
    { \
        const cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error)); \
            exit(1); \
        } \
    }

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 10;
    float A[N], B[N], C[N];

    // Initialize vectors A and B
    for (int i = 0; i < N; ++i) {
        A[i] = i * 1.0f;
        B[i] = i * 2.0f;
    }

    float *d_a, *d_b,*d_c;
    CUDA_CHECK(cudaMalloc(&d_a,N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b,N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c,N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a,A,N*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b,B,N*sizeof(float),cudaMemcpyHostToDevice));

    // --- Debugging: Print device memory before kernel launch ---
    float h_a_debug[N], h_b_debug[N], h_c_debug_before[N];
    CUDA_CHECK(cudaMemcpy(h_a_debug, d_a, N*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b_debug, d_b, N*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c_debug_before, d_c, N*sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Device A before kernel: ";
    for(int i=0; i<N; ++i) std::cout << h_a_debug[i] << " ";
    std::cout << std::endl;

    std::cout << "Device B before kernel: ";
    for(int i=0; i<N; ++i) std::cout << h_b_debug[i] << " ";
    std::cout << std::endl;

    std::cout << "Device C before kernel: ";
    for(int i=0; i<N; ++i) std::cout << h_c_debug_before[i] << " ";
    std::cout << std::endl;
    // --- End Debugging ---


    int blocksize=256;
    int gridsize=(int)ceil((float)N/blocksize); // Cast to float for ceil function
    vectorAdd<<<gridsize,blocksize>>>(d_a,d_b,d_c,N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Add synchronization here

    CUDA_CHECK(cudaMemcpy(C,d_c,N*sizeof(float),cudaMemcpyDeviceToHost));

    // Print the final result
    std::cout << "Final result of vector addition:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0; // Add return statement to main
}
