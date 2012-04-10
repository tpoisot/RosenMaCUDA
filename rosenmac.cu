#include <stdlib.h>
#include <stdio.h>

#define BLOCKS  size
#define THREADS 1
#define T       10000
#define H       0.01
#define R       1.0
#define K       1.0
#define ALPHA   1.0
#define BETA    5.0
#define M       0.2
#define DN      0.0
#define DP      0.05

__global__ void rosmac(float *n0, float *n1, float *p0, float *p1)
{
    // Better integration:
    const int tid = blockIdx.x;
    const int left = (tid == 0)? blockDim.x  - 1 : tid - 1;
    const int right = (tid == blockDim.x - 1)? 0 : tid + 1;
    const float dn = R * n0[tid] * (1.0f - n0[tid] / K) - (ALPHA * n0[tid] * p0[tid]) / (1.0f + BETA * n0[tid]) - DN * (n0[tid] - n0[left] / 2.0f - n0[right] / 2.0f);
    const float dp = (ALPHA * n0[tid] * p0[tid]) / (1.0f + BETA * n0[tid]) - M * p0[tid] - DP * (p0[tid] - p0[left] / 2.0f - p0[right] / 2.0f);
    n1[tid] = n0[tid] + H * dn;
    p1[tid] = p0[tid] + H * dp;
}

int main(int argc, char **argv)
{
    const unsigned int size = (argc == 2)? atof(argv[1]) : 1000;
    const unsigned int bytes = size * sizeof(float);
    float *h_n = (float*)malloc(bytes);
    float *h_p = (float*)malloc(bytes);

    float *d_n0, *d_n1, *d_p0, *d_p1;
    cudaMalloc((void**)&d_n0, bytes);
    cudaMalloc((void**)&d_n1, bytes);
    cudaMalloc((void**)&d_p0, bytes);
    cudaMalloc((void**)&d_p1, bytes);

    // Use gsl:
    srand(42);
    rand();
    for (int i = 0; i < size; ++i)
    {
        h_n[i] = ((float)rand() / RAND_MAX);
        h_p[i] = ((float)rand() / RAND_MAX);
    }

    cudaMemcpy(d_n0, h_n, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_p0, h_p, bytes, cudaMemcpyHostToDevice);

    for (int t = 0; t < T; t += 2)
    {
        rosmac<<<BLOCKS,THREADS>>>(d_n0, d_n1, d_p0, d_p1);
        rosmac<<<BLOCKS,THREADS>>>(d_n1, d_n0, d_p1, d_p0);
        if (t % 100 == 0)
        {
            //printf("%16d -> ", t);
            cudaMemcpy(h_n, d_n1, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_p, d_p1, bytes, cudaMemcpyDeviceToHost);
            for (int i = 0; i < size; ++i)
            {
                printf("%.4f\t", h_p[i]);
            }
            printf("\n");
        }
    }

    cudaFree(d_n0);
    cudaFree(d_n1);
    cudaFree(d_p0);
    cudaFree(d_p1);
    free(h_n);
    free(h_p);
    return EXIT_SUCCESS;
}

