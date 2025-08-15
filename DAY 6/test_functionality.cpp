#include <iostream>

// Small test to verify functionality
#define TEST_WIDTH 4
#define TEST_HEIGHT 3

void transposeMatrix(const float* input, float* output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int inputIndex = y * width + x;
            int outputIndex = x * height + y;
            output[outputIndex] = input[inputIndex];
        }
    }
}

void printMatrix(const float* matrix, int width, int height, const char* name) {
    std::cout << name << " (" << width << "x" << height << "):" << std::endl;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::cout << matrix[y * width + x] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    int width = TEST_WIDTH;
    int height = TEST_HEIGHT;

    // Create test input matrix
    float input[TEST_WIDTH * TEST_HEIGHT] = {
        1, 2, 3, 4,    // Row 0
        5, 6, 7, 8,    // Row 1  
        9, 10, 11, 12  // Row 2
    };
    
    float output[TEST_WIDTH * TEST_HEIGHT];

    // Print original matrix
    printMatrix(input, width, height, "Original Matrix");

    // Transpose
    transposeMatrix(input, output, width, height);

    // Print transposed matrix
    printMatrix(output, height, width, "Transposed Matrix");

    // Verify correctness
    std::cout << "Verification:" << std::endl;
    std::cout << "input[0][1] = " << input[0 * width + 1] << " should equal output[1][0] = " << output[1 * height + 0] << std::endl;
    std::cout << "input[1][2] = " << input[1 * width + 2] << " should equal output[2][1] = " << output[2 * height + 1] << std::endl;
    std::cout << "input[2][3] = " << input[2 * width + 3] << " should equal output[3][2] = " << output[3 * height + 2] << std::endl;

    return 0;
}
