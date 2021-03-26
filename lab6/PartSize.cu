#include <iostream>


using namespace std;



#define CUDA_CHECK_RETURN(value) {\
    cudaError_t _m_cudaStat = value;\
    if (_m_cudaStat != cudaSuccess) {\
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
        exit(1);\
    }}



__global__ void VecSum(float *A, float *B, float *C, int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < size)
        C[i] = A[i] + B[i];
}



__global__ void VecMul(float *A, float *B, float *C, int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < size)
        C[i] = A[i] * B[i];
}



int main(int argc, char* argv[])
{
    if (argc != 3) {
        cout << "launch parametrs: [vector size] [partSize]" << endl;
        return 1;
    }

    cudaDeviceProp prop;
    int whichDevice;

    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (!prop.deviceOverlap) {
        cout << "Device does not support overlapping" << endl;
        return 1;
    }

    int full_data_size = atoi(argv[1]);
    int partSize = atoi(argv[2]);

    if (full_data_size % partSize != 0) {
        cout << "The size of the data chunk must be a multiple of the full data size" << endl;
        return 1;
    }

    float *A = new float[full_data_size];
    float *B = new float[full_data_size];
    float *C = new float[full_data_size];

    float *dev_a, *dev_b, *dev_c;
    cudaHostAlloc((void**)&dev_a, full_data_size * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&dev_b, full_data_size * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&dev_c, full_data_size * sizeof(int), cudaHostAllocDefault);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int i = 0; i < full_data_size; i += partSize) {
        cudaMemcpyAsync(dev_a, A + i, partSize * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dev_b, B + i, partSize * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dev_c, C + i, partSize * sizeof(int), cudaMemcpyHostToDevice, stream);
        VecMul <<< partSize / 256, 256, 0, stream >>> (dev_a, dev_b, dev_c, full_data_size);
        cudaMemcpyAsync(C + i, dev_c, partSize * sizeof(int), cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "time: " << elapsedTime << " ms" << endl;

    return 0;
}
