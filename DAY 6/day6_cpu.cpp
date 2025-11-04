      include <iostream>
#include <chrono>

// Define the size of the matrix
#define WIDTH 1024
#define HEIGHT 1024

// CPU function for matrix transposition
void transposeMatrix(const float* input, float* output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int inputIndex = y * width + x;
            int outputIndex = x * height + y;
            output[outputIndex] = input[inputIndex];
        }
    }
}

int main() {
    int width = WIDTH;
    int height = HEIGHT;

    // Allocate host memory
    size_t size = width * height * sizeof(float);
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);

    // Initialize the input matrix with some values
    for (int i = 0; i < width * height; i++) {
        h_input[i] = static_cast<float>(i);
    }

    std::cout << "Starting matrix transposition on CPU..." << std::endl;
    
    // Time the transposition
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform the transposition
    transposeMatrix(h_input, h_output, width, height);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Verify the result
    bool success = true;
    for (int i = 0; i < width && success; i++) {
        for (int j = 0; j < height && success; j++) {
            if (h_output[i * height + j] != h_input[j * width + i]) {
                success = false;
            }
        }
    }

    std::cout << (success ? "Matrix transposition succeeded!" : "Matrix transposition failed!") << std::endl;
    std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}
