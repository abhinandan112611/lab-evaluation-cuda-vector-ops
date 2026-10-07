#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    printf("CUDA error: %s (line %d)\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void vecMul(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

int main(int argc, char** argv) {
    int n       = (argc > 1) ? atoi(argv[1]) : 1000000;   // vector size
    int threads = (argc > 2) ? atoi(argv[2]) : 256;       // threads per block
    bool isMul  = (argc > 3) && (atoi(argv[3]) == 1);     // 0 = add, 1 = multiply
    size_t bytes = (size_t)n * sizeof(float);

    // ---- Host memory ----
    float *h_a = (float*)malloc(bytes), *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes), *h_ref = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // ---- CPU (sequential) ----
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++)
        h_ref[i] = isMul ? h_a[i] * h_b[i] : h_a[i] + h_b[i];
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---- Device memory ----
    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_c, bytes));

    int blocks = (n + threads - 1) / threads;

    // Warm-up launch (the first CUDA call is slow because of initialisation)
    vecAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, k0, k1, stop;
    cudaEventCreate(&start); cudaEventCreate(&k0);
    cudaEventCreate(&k1);    cudaEventCreate(&stop);

    // ---- GPU: copy in, kernel, copy out ----
    cudaEventRecord(start);
    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(k0);
    if (isMul) vecMul<<<blocks, threads>>>(d_a, d_b, d_c, n);
    else       vecAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaEventRecord(k1);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_ms, total_ms;
    cudaEventElapsedTime(&kernel_ms, k0, k1);
    cudaEventElapsedTime(&total_ms, start, stop);

    // ---- Verify ----
    int errors = 0;
    for (int i = 0; i < n; i++)
        if (fabsf(h_c[i] - h_ref[i]) > 1e-5f) errors++;

    // Effective bandwidth: 2 reads + 1 write = 3 * bytes
    double gbps = (3.0 * bytes) / (kernel_ms * 1e-3) / 1e9;

    printf("op=%s,N=%d,threads=%d,cpu_ms=%.3f,gpu_kernel_ms=%.3f,gpu_total_ms=%.3f,"
           "speedup_kernel=%.2f,speedup_total=%.2f,bandwidth_GBps=%.1f,errors=%d\n",
           isMul ? "mul" : "add", n, threads, cpu_ms, kernel_ms, total_ms,
           cpu_ms / kernel_ms, cpu_ms / total_ms, gbps, errors);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c); free(h_ref);
    return 0;
}