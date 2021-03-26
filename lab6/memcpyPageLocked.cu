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
    cudaDeviceProp prop;
    int whichDevice;

    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (!prop.deviceOverlap) {
        cout << "Device does not support overlapping" << endl;
        return 0;
    }

    int *host_a, *host_b, *host_c;
    int *dev_a, *dev_b, *dev_c;

    cudaHostAlloc((void**)&dev_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&dev_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&dev_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);

    cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int i = 0; i < FULL_DATA_SIZE; i += N) {
        cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
        kernel <<< N / 256, 256, 0, stream >>> (dev_a, dev_b, dev_c);
        cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "time: " << elapsedTime << " ms" << endl;
}
