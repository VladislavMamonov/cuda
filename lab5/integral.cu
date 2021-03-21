#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define M_PI 3.14159265358979323846
#define COEF 48
#define VERTCOUNT COEF*COEF*2
#define RADIUS 10.0f
#define FGSIZE 20
#define FGSHIFT FGSIZE / 2
#define IMIN(A, B) (A < B ? A : B)
#define THREADSPERBLOCK 256
#define BLOCKSPERGRID IMIN(32, (VERTCOUNT + THREADSPERBLOCK - 1) / THREADSPERBLOCK)


typedef float(*ptr_f)(float, float, float);

struct Vertex {
    float x, y, z;
};

__constant__ Vertex vert[VERTCOUNT];


float func(float x, float y, float z)
{
    return (0.5 * sqrtf(15.0 / M_PI)) * (0.5 * sqrtf(15.0 / M_PI))
        * z * z * y * y * sqrtf(1.0f - z * z / RADIUS / RADIUS) / RADIUS / RADIUS / RADIUS / RADIUS;
}



float check(Vertex *v, ptr_f f)
{
    float sum = 0.0f;

    for (int i = 0; i < VERTCOUNT; ++i)
        sum += f(v[i].x, v[i].y, v[i].z);

    return sum;
}



void calc_f(float *arr_f, int x_size, int y_size, int z_size, ptr_f f)
{
    for (int x = 0; x < x_size; ++x)
        for (int y = 0; y < y_size; ++y)
            for (int z = 0; z < z_size; ++z)
                arr_f[z_size * (x * y_size + y) + z] = f(x - FGSHIFT, y - FGSHIFT, z - FGSHIFT);
}



void init_vertices()
{
    Vertex *temp_vert = (Vertex *)malloc(sizeof(Vertex) * VERTCOUNT);
    int i = 0;
    for (int iphi = 0; iphi < 2 * COEF; ++iphi) {
        for (int ipsi = 0; ipsi < COEF; ++ipsi, ++i) {
            float phi = iphi * M_PI / COEF;
            float psi = ipsi * M_PI / COEF;
            temp_vert[i].x = RADIUS * sinf(psi) * cosf(phi);
            temp_vert[i].y = RADIUS * sinf(psi) * sinf(phi);
            temp_vert[i].z = RADIUS * cosf(psi);
        }
    }

    printf("sumcheck = %f\n", check(temp_vert, &func) * M_PI * M_PI / COEF / COEF);
    cudaMemcpyToSymbol(vert, temp_vert, sizeof(Vertex) * VERTCOUNT, 0, cudaMemcpyHostToDevice);

    free(temp_vert);
}


__device__ float Trilinear_Interpolation(float x, float y, float z, float *arr) {   //Трилинейная интерполяция
    float res = 0.0f;
    float x_r[2], y_r[2], z_r[2];
    x_r[0] = x - 1;
    x_r[1] = x + 1;
    y_r[0] = y - 1;
    y_r[1] = y + 1;
    z_r[0] = z - 1;
    z_r[1] = z + 1;
    float denominator = (x_r[1] - x_r[0]) * (y_r[1] - y_r[0]) * (z_r[1] - z_r[0]);
    res += (arr[FGSIZE * ((int)x_r[0] * FGSIZE + (int)y_r[0]) + (int)z_r[0]] * (x_r[1] - x) * (y_r[1] - y) * (z_r[1] - z)) / denominator;
    res += (arr[FGSIZE * ((int)x_r[0] * FGSIZE + (int)y_r[0]) + (int)z_r[1]] * (x_r[1] - x) * (y_r[1] - y) * (z - z_r[0])) / denominator;
    res += (arr[FGSIZE * ((int)x_r[0] * FGSIZE + (int)y_r[1]) + (int)z_r[0]] * (x_r[1] - x) * (y - y_r[0]) * (z_r[1] - z)) / denominator;
    res += (arr[FGSIZE * ((int)x_r[0] * FGSIZE + (int)y_r[1]) + (int)z_r[1]] * (x_r[1] - x) * (y - y_r[0]) * (z - z_r[0])) / denominator;
    res += (arr[FGSIZE * ((int)x_r[1] * FGSIZE + (int)y_r[0]) + (int)z_r[0]] * (x - x_r[0]) * (y_r[1] - y) * (z_r[1] - z)) / denominator;
    res += (arr[FGSIZE * ((int)x_r[1] * FGSIZE + (int)y_r[0]) + (int)z_r[1]] * (x - x_r[0]) * (y_r[1] - y) * (z - z_r[0])) / denominator;
    res += (arr[FGSIZE * ((int)x_r[1] * FGSIZE + (int)y_r[1]) + (int)z_r[0]] * (x - x_r[0]) * (y - y_r[0]) * (z_r[1] - z)) / denominator;
    res += (arr[FGSIZE * ((int)x_r[1] * FGSIZE + (int)y_r[1]) + (int)z_r[1]] * (x - x_r[0]) * (y - y_r[0]) * (z - z_r[0])) / denominator;
    //printf("%f\n", res);
    return res;
}


__global__ void kernel(float *a, float *val)
{
    __shared__ float cache[THREADSPERBLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float x = vert[tid].x + FGSHIFT + 0.5f;
    float y = vert[tid].y + FGSHIFT + 0.5f;
    float z = vert[tid].z + FGSHIFT + 0.5f;
    cache[cacheIndex] = Trilinear_Interpolation(x, y, z, val);

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (cacheIndex < s)
            cache[cacheIndex] += cache[cacheIndex + s];
        __syncthreads();
    }

    if (cacheIndex == 0)
        a[blockIdx.x] = cache[0];
}



int main()
{
    float *arr = (float *)malloc(sizeof(float) * FGSIZE * FGSIZE * FGSIZE);
    float *sum = (float*)malloc(sizeof(float) * BLOCKSPERGRID);
    float *sum_dev;
    float *values;
    init_vertices();
    calc_f(arr, FGSIZE, FGSIZE, FGSIZE, &func);

    cudaMalloc((void**)&sum_dev, sizeof(float) * BLOCKSPERGRID);
    cudaMalloc((void**)&values, sizeof(float) * FGSIZE * FGSIZE * FGSIZE);
    cudaMemcpy(values, arr, sizeof(float) * FGSIZE * FGSIZE * FGSIZE, cudaMemcpyHostToDevice);

    kernel <<< BLOCKSPERGRID,THREADSPERBLOCK >>> (sum_dev, values);
    cudaDeviceSynchronize();
    cudaMemcpy(sum, sum_dev, sizeof(float) * BLOCKSPERGRID, cudaMemcpyDeviceToHost);

    float s = 0.0f;
    for (int i = 0; i < BLOCKSPERGRID; ++i)
        s += sum[i];
    printf("sum = %f\n", s * M_PI * M_PI / COEF / COEF);

    cudaFree(sum_dev);
    free(sum);
    free(arr);

    return 0;
}
