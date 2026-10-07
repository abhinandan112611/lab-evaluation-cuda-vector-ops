\# GPU Vector Operations using CUDA (Team 7)



Parallel Computing Mini-Project: Lab Evaluation



\## Team Members

\- \[Name 1, Roll No.]

\- \[Name 2, Roll No.]

\- \[Name 3, Roll No.]



\## Objective

Perform vector addition and element-wise multiplication using CUDA threads

and compare the execution time against a sequential CPU implementation.



\## Hardware and Software

\- GPU: NVIDIA GeForce GTX 1650 Ti (Laptop), 4 GB

\- Driver: 617.42, CUDA Toolkit 13.4

\- Compiler: nvcc with Visual Studio 2026 (MSVC), Windows 11

\- Analysis: Python 3.13, pandas, matplotlib



\## Method

\- Sequential: a single CPU loop computing c\[i] = a\[i] + b\[i] (or a\[i] \* b\[i]).

\- Parallel: one CUDA thread per element, with a bounds check and

&#x20; blocks = (N + threads - 1) / threads.

\- Timing: std::chrono for the CPU, cudaEvent for the GPU. Kernel-only time

&#x20; and total time (host-to-device copy + kernel + device-to-host copy) are

&#x20; recorded separately.

\- Correctness: every GPU result is compared with the CPU result (errors = 0 in all runs).

\- Experiments: N = 10^4, 10^5, 10^6, 10^7, 5x10^7; block sizes 64 to 1024;

&#x20; both operations; 5 repeats per setting (250 runs), averaged.

\- Efficiency = achieved memory bandwidth / peak bandwidth (\[192] GB/s assumed).



\## Repository Structure

\- src/      CUDA source (vec\_gpu.cu) and Python scripts

\- results/  raw timings (results.csv), averaged.csv, summary.csv

\- graphs/   execution time, speedup, efficiency and block-size plots



\## How to Build and Run

Open "x64 Native Tools Command Prompt for VS" and go to the project folder.



&#x20;   nvcc -O2 src\\vec\_gpu.cu -o vec\_gpu.exe

&#x20;   vec\_gpu.exe 1000000 256 0        (arguments: N, threads per block, 0=add 1=mul)

&#x20;   python src\\run\_experiments.py    (runs all 250 experiments)

&#x20;   python src\\make\_graphs.py        (averages results, makes graphs)



Run all commands from the project root folder.



\## Results Summary (256 threads per block, vector add)



| N | CPU (ms) | GPU kernel (ms) | GPU total (ms) | Kernel speedup | Total speedup | Efficiency |

|---|---|---|---|---|---|---|

| 10^4 | 0.013 | 0.020 | 0.173 | 1.05x | 0.07x | 5.2% |

| 10^5 | 0.089 | 0.044 | 0.597 | 2.06x | 0.15x | 14.4% |

| 10^6 | 1.659 | 0.107 | 4.217 | 15.6x | 0.40x | 58.7% |

| 10^7 | 14.346 | 0.741 | 31.801 | 19.4x | 0.45x | 84.5% |

| 5x10^7 | 73.356 | 3.481 | 151.965 | 21.1x | 0.48x | 89.8% |



Multiplication behaves almost identically.



\## Key Findings

\- The GPU kernel is up to about 21x faster than the sequential CPU loop at large N.

\- Including PCIe transfers, the GPU is slower end to end (total speedup below 1x),

&#x20; because vector add and multiply do very little work per byte moved.

\- At small N, launch overhead dominates and the GPU gives little or no gain.

\- Efficiency rises to about 90% of peak memory bandwidth at large N.

\- Block size (64 to 1024) has almost no effect, since the kernels are memory-bound.

\- GPU and CPU results match exactly in all 250 runs.

