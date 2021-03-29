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



__global__ void VecMul(float *A, float *B, float *C, int partSize)
{
    __shared__ float cache[256];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    int temp = 0;

    while (tid < partSize) {
        temp += A[tid] * B[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (cacheIndex < s) {
            cache[cacheIndex] += cache[cacheIndex + s];
        }
        __syncthreads();
    }

    if (cacheIndex == 0) C[blockIdx.x] = cache[0];
}


void InitVec(float *A, float *B, int size)
{
    for (int i = 0; i < size; i++) {
        A[i] = 1;
        B[i] = 1;
    }
}


void printVec(float *vec, int size)
{
    for (int i = 0; i < size; i++)
        cout << vec[i] << endl;
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

    float *dev_a, *dev_b, *dev_c;
    float *A, *B, *C;

    cudaHostAlloc((void**)&A, full_data_size * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&B, full_data_size * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&C, full_data_size * sizeof(float), cudaHostAllocDefault);

    InitVec(A, B, full_data_size);

    cudaMalloc((void**)&dev_a, full_data_size * sizeof(float));
    cudaMalloc((void**)&dev_b, full_data_size * sizeof(float));
    cudaMalloc((void**)&dev_c, full_data_size * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int i = 0; i < full_data_size; i += partSize) {
        cudaMemcpyAsync(dev_a, A + i, partSize * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dev_b, B + i, partSize * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dev_c, C + i, partSize * sizeof(float), cudaMemcpyHostToDevice, stream);
        VecSum <<< partSize / 256, 256, 0, stream >>> (dev_a, dev_b, dev_c, full_data_size);
        cudaMemcpyAsync(C + i, dev_c, partSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }

    //printVec(C, full_data_size);

    cudaStreamSynchronize(stream);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "time: " << elapsedTime << " ms" << endl;

    return 0;
}
