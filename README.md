# Approximate Pattern Matching
<!--toc:start-->
  - [Build && Run](#build-run)
  - [Test](#test)
  - [Explanation](#explanation)
- [MPI Parallelization for Approximate Pattern Matching](#mpi-parallelization-for-approximate-pattern-matching)
  - [Parallelization Strategy](#parallelization-strategy)
    - [1. Text Partitioning](#1-text-partitioning)
    - [2. Handling Overlapping Boundaries](#2-handling-overlapping-boundaries)
    - [3. Parallel Computation of Matches](#3-parallel-computation-of-matches)
    - [4. Gathering Results](#4-gathering-results)
  - [Performance Benefits](#performance-benefits)
- [OpenMP Optimization for Approximate Pattern Matching](#openmp-optimization-for-approximate-pattern-matching)
  - [Parallelization Strategy](#parallelization-strategy)
    - [Different OpenMP Configurations Tested](#different-openmp-configurations-tested)
  - [Results and Observations](#results-and-observations)
  - [Conclusion](#conclusion)
- [CUDA and OpenMP Optimization for Approximate Pattern Matching](#cuda-and-openmp-optimization-for-approximate-pattern-matching)
  - [**CUDA Optimization Strategy**](#cuda-optimization-strategy)
    - [**1. Parallelizing Text Processing on the GPU**](#1-parallelizing-text-processing-on-the-gpu)
    - [**2. Memory Management and Data Transfer Optimization**](#2-memory-management-and-data-transfer-optimization)
    - [**3. CUDA Kernel Execution Model**](#3-cuda-kernel-execution-model)
  - [**Merging CUDA with OpenMP**](#merging-cuda-with-openmp)
    - [**1. Parallelizing Kernel Launches**](#1-parallelizing-kernel-launches)
    - [**2. OpenMP's Impact on Performance**](#2-openmps-impact-on-performance)
    - [**3. Avoiding Race Conditions**](#3-avoiding-race-conditions)
  - [**Performance Improvements and Scalability**](#performance-improvements-and-scalability)
    - [**Efficient GPU Utilization**](#efficient-gpu-utilization)
    - [**Optimized Memory Access**](#optimized-memory-access)
    - [**Hybrid CPU-GPU Execution (Limited OpenMP Impact)**](#hybrid-cpu-gpu-execution-limited-openmp-impact)
  - [**Conclusion**](#conclusion)
    - [**Future Optimizations**](#future-optimizations)
<!--toc:end-->

## Build && Run

```bash
source set_env.sh
make
OMP_NUM_THREADS=12 salloc -N 1 -n 6 mpirun ./apm_cuda 25 ./dna/big_chrY.fa CCAGTTCCCTTCTGGAATTTAGGGGCCCTGGGACAGCCCTGTACATGAGC CATACCGATAACAACCACGAGCTAGTAAGCGCCGTCGCGCCAATAAATCT ACCCTCATTGGTCAGGTCCAGCGCATAGGGTAGGATAGGATCTGTACCAT
```

## Test

Runs on `big_chrY.fa` and 3 patterns.

```bash
source set_env.sh
make
cd test
chmod +x apm_*
python3 runner.py
```

Output:

```bash

Running Base Implementation...

Running Optimized Implementation...

===================================
          COMPARISON RESULTS       
===================================

Base Implementation Time: 59.9952 seconds
Optimized Implementation Time: 1.6439 seconds

✅ Results Match! The optimization is correct.

-----------------------------------
| Pattern | Base Matches | Optimized Matches |
-----------------------------------
| CCAGTTCCCTTCTGGAATTTAGGGGCCCTGGGACAGCCCTGTACATGAGC | 130016       | 130016            |
| CATACCGATAACAACCACGAGCTAGTAAGCGCCGTCGCGCCAATAAATCT | 224129       | 224129            |
| TATGCCACATGCCCGGAATTAGGTCTGTTACTCGTAGCAAACGTATGCGG | 289548       | 289548            |
-----------------------------------

Speedup Factor: 36.5
```
