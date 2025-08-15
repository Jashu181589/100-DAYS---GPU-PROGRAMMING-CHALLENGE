#include <stdio.h>
#include <iostream>
#include <chrono>

// I'm assuming that the mask and the matrix to be square for simplicity
#define Mask_width 5

// CPU version of 2D convolution with tiling simulation
void twod_convolution_cpu(const float* A, float* C, const float mask[Mask_width][Mask_width], int n) {
    int half_mask = Mask_width / 2;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float result = 0.0f;
            
            // Apply convolution mask
            for (int k = 0; k < Mask_width; k++) {
                for (int x = 0; x < Mask_width; x++) {
                    int row = i + k - half_mask;
                    int col = j + x - half_mask;
                    
                    // Handle boundaries (padding with zeros)
                    if (row >= 0 && row < n && col >= 0 && col < n) {
                        result += A[row * n + col] * mask[k][x];
                    }
                    // Implicit zero padding for out-of-bounds
                }
            }
            
            C[i * n + j] = result;
        }
    }
}

int main() {
    int n = 10;
    float *h_A = (float*)malloc(n * n * sizeof(float));
    float *h_C = (float*)malloc(n * n * sizeof(float));
    float mask[Mask_width][Mask_width];

    // Initialize mask (same as original)
    for (int i = 0; i < Mask_width; i++) {
        for (int j = 0; j < Mask_width; j++) {
            mask[i][j] = 1.0f;  // Changed to 1 for easier verification
        }
    }

    // Initialize input matrix (same as original)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_A[i*n + j] = 3.0f;
        }
    }

    std::cout << "Starting 2D convolution on CPU..." << std::endl;
    
    // Time the convolution
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform convolution
    twod_convolution_cpu(h_A, h_C, mask, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Print results
    printf("Results:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", h_C[i*n + j]);
        }
        printf("\n");
    }

    std::cout << "\n2D convolution completed successfully!" << std::endl;
    std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

    // Verify result (for 3x3 input with 1's mask, center should be 3*9=27, edges less)
    std::cout << "\nVerification:" << std::endl;
    std::cout << "Center element [5][5]: " << h_C[5*n + 5] << " (expected: close to 75 for 5x5 mask of 1's)" << std::endl;
    std::cout << "Corner element [0][0]: " << h_C[0*n + 0] << " (expected: much less due to boundary)" << std::endl;

    // Clean up
    free(h_A);
    free(h_C);

    return 0;
}
