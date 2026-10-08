<h1 align="center">GPU Vector Operations using CUDA</h1>

<p align="center">
  Vector addition and element-wise multiplication on the GPU, benchmarked against a sequential CPU baseline.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-13.4-76B900?logo=nvidia&logoColor=white" alt="CUDA 13.4">
  <img src="https://img.shields.io/badge/C%2B%2B-nvcc-00599C?logo=cplusplus&logoColor=white" alt="C++">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white" alt="Python 3.13">
  <img src="https://img.shields.io/badge/GPU-GTX%201650%20Ti-76B900?logo=nvidia&logoColor=white" alt="GTX 1650 Ti">
</p>

**Course:** Parallel Computing Mini-Project (Lab Evaluation), Team 7: GPU Vector Operations (CUDA)

## Team Members

1. Abhinandan Patil
2. Preetam N
3. Sai Aaryan
4. Amit K

---

## Table of Contents

1. [Objective](#objective)
2. [Environment](#environment)
3. [Method](#method)
4. [Repository Structure](#repository-structure)
5. [Build and Run](#build-and-run)
6. [Execution Screenshots](#execution-screenshots)
7. [Results](#results)
8. [Graphs](#graphs)
9. [Analysis](#analysis)
10. [Limitations](#limitations)

---

## Objective

Implement vector addition and element-wise multiplication using CUDA threads, and compare execution time against a sequential CPU implementation across different vector sizes and block sizes.

## Environment

| Component | Details |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 Ti (Laptop), 4 GB |
| Driver | 617.42 |
| CUDA Toolkit | 13.4 |
| Compiler | nvcc with Visual Studio 2026 (MSVC) |
| OS | Windows 11 |
| Analysis | Python 3.13, pandas, matplotlib |

## Method

- **Sequential baseline:** a single CPU loop computing `c[i] = a[i] + b[i]` (or `a[i] * b[i]`).
- **Parallel version:** one CUDA thread per element, with a bounds check `if (i < n)` and `blocks = (N + threads - 1) / threads`. Each thread finds its element with `i = blockIdx.x * blockDim.x + threadIdx.x`.
- **Data flow:** vectors are generated in host RAM, copied to GPU memory over PCIe (`cudaMemcpy`), processed by the kernel, and the result is copied back.
- **Timing:** `std::chrono` for the CPU and `cudaEvent` for the GPU. Two GPU times are recorded:
  - *Kernel only*: kernel execution.
  - *Total*: host-to-device copy + kernel + device-to-host copy.
- **Warm-up:** one untimed kernel launch before measuring, to exclude CUDA initialisation cost.
- **Correctness:** every GPU result is compared element by element with the CPU result.
- **Experiments:** N = 10⁴, 10⁵, 10⁶, 10⁷, 5×10⁷; block sizes 64, 128, 256, 512, 1024; both operations; 5 repeats per setting (250 runs), averaged.
- **Efficiency:** achieved memory bandwidth ÷ assumed peak bandwidth (about 192 GB/s), where achieved bandwidth = 3 × N × 4 bytes ÷ kernel time.

## Repository Structure

```
lab-evaluation-cuda-vector-ops/
├── README.md
├── src/
│   ├── vec_gpu.cu           CUDA program (CPU baseline + GPU kernels + timing)
│   ├── run_experiments.py   runs all 250 experiments, writes results.csv
│   ├── make_graphs.py       averages repeats, produces graphs and summary
│   └── check.py             sanity check on results.csv
├── results/
│   ├── results.csv          raw timings (250 rows)
│   ├── averaged.csv         averages over the 5 repeats
│   └── summary.csv          table at 256 threads per block
├── graphs/                  execution time, speedup, efficiency, block-size plots
├── screenshots/             terminal screenshots of the execution
└── presentation/            lab evaluation slides
```

## Build and Run

Open **x64 Native Tools Command Prompt for VS**, go to the project root, then:

```
nvcc -O2 src\vec_gpu.cu -o vec_gpu.exe
vec_gpu.exe 1000000 256 0
python src\run_experiments.py
python src\make_graphs.py
```

`vec_gpu.exe` takes three arguments: vector size `N`, threads per block, and operation (`0` = add, `1` = multiply). Run all commands from the project root.

## Execution Screenshots

### 1. GPU and driver check

`nvidia-smi` confirms the GTX 1650 Ti with driver 617.42 and CUDA 13.4.

<img src="screenshots/01-nvidia-smi-driver-617.42.png" alt="nvidia-smi output" width="750">

### 2. CUDA compiler check

`nvcc --version` confirms CUDA toolkit 13.4.

<img src="screenshots/02-nvcc-version.png" alt="nvcc version output" width="520">

### 3. First CUDA test: four GPU threads

A small test program compiled with `nvcc` and run on the GPU. Four threads print their IDs, which confirms the toolchain works.

<img src="screenshots/03-gpu-hello-test.png" alt="Hello from GPU thread test" width="650">

### 4. Compile and first runs of the vector program

`vec_gpu.exe` run for addition and multiplication at N = 10⁶, and addition at N = 5×10⁷. Every run ends with `errors=0`, so the GPU result matches the CPU result.

<img src="screenshots/04-compile-and-first-runs.png" alt="Compile and first runs" width="900">

### 5. Running all 250 experiments

`run_experiments.py` runs every combination of operation, vector size and block size, 5 repeats each, and finishes with `ALL DONE`.

<img src="screenshots/05-run-all-250-experiments.png" alt="Experiment run completed" width="520">

### 6. Validating the results file

`check.py` confirms that all 250 rows were recorded and that no run had a mismatch.

<img src="screenshots/06-validate-results-csv.png" alt="check.py output" width="420">

### 7. Generating graphs and the summary table

`make_graphs.py` averages the 5 repeats, computes efficiency, saves the graphs and prints the summary table.

<img src="screenshots/07-generate-graphs-and-summary.png" alt="make_graphs.py output" width="900">

### 8. Publishing to GitHub

The project committed and pushed to this repository.

<img src="screenshots/08-push-to-github.png" alt="git push output" width="800">

## Results

Averages of 5 runs at 256 threads per block. Kernel speedup = CPU time ÷ GPU kernel time. Total speedup = CPU time ÷ GPU total time.

### Vector addition

| N | CPU (ms) | GPU kernel (ms) | GPU total (ms) | Kernel speedup | Total speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 10⁴ | 0.013 | 0.020 | 0.173 | 1.05× | 0.07× | 5.2% |
| 10⁵ | 0.089 | 0.044 | 0.597 | 2.06× | 0.15× | 14.4% |
| 10⁶ | 1.659 | 0.107 | 4.217 | 15.6× | 0.40× | 58.7% |
| 10⁷ | 14.346 | 0.741 | 31.801 | 19.4× | 0.45× | 84.5% |
| 5×10⁷ | 73.356 | 3.481 | 151.965 | 21.1× | 0.48× | 89.8% |

### Vector multiplication

| N | CPU (ms) | GPU kernel (ms) | GPU total (ms) | Kernel speedup | Total speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 10⁴ | 0.013 | 0.040 | 0.190 | 0.38× | 0.07× | 1.8% |
| 10⁵ | 0.081 | 0.049 | 0.579 | 1.65× | 0.14× | 12.7% |
| 10⁶ | 1.583 | 0.114 | 3.419 | 13.9× | 0.46× | 54.8% |
| 10⁷ | 13.603 | 0.744 | 29.659 | 18.3× | 0.46× | 84.1% |
| 5×10⁷ | 68.233 | 3.480 | 151.333 | 19.6× | 0.45× | 89.8% |

GPU and CPU results matched exactly in all 250 runs (0 errors).

## Graphs

### Vector addition

<table>
  <tr>
    <td align="center"><b>Execution time vs N</b><br><img src="graphs/time_vs_N_add.png" width="420"></td>
    <td align="center"><b>Speedup vs N</b><br><img src="graphs/speedup_vs_N_add.png" width="420"></td>
  </tr>
  <tr>
    <td align="center"><b>Efficiency vs N</b><br><img src="graphs/efficiency_vs_N_add.png" width="420"></td>
    <td align="center"><b>Block size vs kernel time</b><br><img src="graphs/blocksize_add.png" width="420"></td>
  </tr>
</table>

### Vector multiplication

<table>
  <tr>
    <td align="center"><b>Execution time vs N</b><br><img src="graphs/time_vs_N_mul.png" width="420"></td>
    <td align="center"><b>Speedup vs N</b><br><img src="graphs/speedup_vs_N_mul.png" width="420"></td>
  </tr>
  <tr>
    <td align="center"><b>Efficiency vs N</b><br><img src="graphs/efficiency_vs_N_mul.png" width="420"></td>
    <td align="center"><b>Block size vs kernel time</b><br><img src="graphs/blocksize_mul.png" width="420"></td>
  </tr>
</table>

## Analysis

- **The GPU kernel is up to about 21× faster** than the sequential CPU loop at large N.
- **End to end, the GPU is slower** (total speedup below 1×). Vector add and multiply do very little work per byte, so the PCIe transfers cost more than the computation saves.
- **Small vectors do not benefit.** At N = 10⁴, kernel launch overhead is as long as the whole job, and the CPU is faster. The kernel overtakes the CPU between N = 10⁴ and 10⁵.
- **Efficiency reaches about 90% of peak memory bandwidth** at large N, so the kernels are memory-bound. This is also why addition and multiplication perform almost identically.
- **Block size has almost no effect** (64 to 1024 threads), which is expected for memory-bound kernels.
- **The GPU would pay off overall** if the data stayed on the device across several operations (roughly three or more on the same data, estimated from these results), or if each element needed heavier computation.

## Limitations

- Measurements at N = 10⁴ and 10⁵ are noisy because the kernels run for only microseconds.
- The efficiency figure assumes a peak bandwidth of about 192 GB/s. The exact value should be checked against the GPU specification.
- The largest vector size was capped at 5×10⁷ to stay safely within the 4 GB of GPU memory, which was shared with the display and other applications. By that size, speedup and efficiency had already levelled off.
- The CPU baseline is single-threaded. A multi-threaded (OpenMP) baseline would narrow the gap.
- Results come from a single laptop GPU running Windows with background applications.
