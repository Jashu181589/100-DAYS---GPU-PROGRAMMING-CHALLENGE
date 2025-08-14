#include <stdio.h>

// Simple CPU implementation of prefix sum
void prefixSumCPU(int *input, int *output, int n) {
    output[0] = input[0];
    for (int i = 1; i < n; i++) {
        output[i] = output[i-1] + input[i];
    }
}

int main() {
    const int N = 16;

    int h_input[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int h_output[N];
    
    // Calculate prefix sum on CPU
    prefixSumCPU(h_input, h_output, N);
    
    // Print input and output
    printf("Input: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_input[i]);
    }
    
    printf("\nOutput (Prefix Sum): ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    return 0;
}
