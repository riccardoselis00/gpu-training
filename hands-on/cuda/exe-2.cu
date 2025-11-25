#include <cstdio>
#include <cmath>

__global__
void vec_add(int n, const float *A, const float *B, float *C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // TODO 1: implementa C[i] = A[i] + B[i];
        /* TODO 1 */
        // SOLUZIONE TODO 1: C[i] = A[i] + B[i];
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1 << 20;  // ~1M elementi
    size_t size = N * sizeof(float);

    // Allocazione host
    float *h_A = (float*) malloc(size);
    float *h_B = (float*) malloc(size);
    float *h_C = (float*) malloc(size);

    // Inizializzazione
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // TODO 2: allocazione device
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    // SOLUZIONE TODO 2:
    // cudaMalloc(&d_A, size);
    // cudaMalloc(&d_B, size);
    // cudaMalloc(&d_C, size);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // TODO 3: copia A e B da host a device
    // SOLUZIONE TODO 3:
    // cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Configurazione kernel (es. 256 thread per blocco)
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vec_add<<<blocksPerGrid, threadsPerBlock>>>(N, d_A, d_B, d_C);
    cudaDeviceSynchronize();

    // TODO 4: copia C da device a host
    // SOLUZIONE TODO 4:
    // cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verifica risultato
    float maxError = 0.0f;
    for (int i = 0; i < N; ++i) {
        maxError = fmaxf(maxError, fabsf(h_C[i] - 3.0f));
    }
    printf("Max error = %f\n", maxError);

    // Free
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}