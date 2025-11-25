#include <cstdio>

__global__
void fill_indices(int *data, int n) {
    int i =  blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < n) {
        data[i] = i;
    }
}

int main() {
    const int N = 16;
    int h_data[N];

    int *d_data = nullptr;
    cudaMalloc(&d_data, N * sizeof(int));

    // lanciamo 1 blocco da 16 thread (per iniziare)
    dim3 blockSize(16);
    dim3 gridSize(1);

    fill_indices<<<gridSize, blockSize>>>(d_data, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Risultato:\n");
    for (int i = 0; i < N; ++i) {
        printf("h_data[%d] = %d\n", i, h_data[i]);
    }

    cudaFree(d_data);
    return 0;
}
