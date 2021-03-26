#include <iostream>


#define N (1024 * 1024)
#define FULL_DATA_SIZE (N * 20)


using namespace std;



#define CUDA_CHECK_RETURN(value) {\
    cudaError_t _m_cudaStat = value;\
    if (_m_cudaStat != cudaSuccess) {\
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
        exit(1);\
    }}



__global__ void kernel(int *a, int *b, int *c)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}



int main()
{
    int *dev_a, *dev_b, *dev_c;

    int *host_a = new int[FULL_DATA_SIZE];
    int *host_b = new int[FULL_DATA_SIZE];
    int *host_c = new int[FULL_DATA_SIZE];

    cudaMalloc((void**)&dev_a, FULL_DATA_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_b, FULL_DATA_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_c, FULL_DATA_SIZE * sizeof(int));

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int i = 0; i < FULL_DATA_SIZE; i += N) {
        cudaMemcpy(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice);
        kernel <<< N / 256, 256, 0 >>> (dev_a, dev_b, dev_c);
        cudaMemcpy(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "time: " << elapsedTime << " ms" << endl;
}
