# Comparative Analysis of Parallel & Distributed Computing Paradigms

## 1. Executive Summary

This project presents a comparative benchmarking study of dense $4000 \times 4000$ Matrix Multiplication ($C = A \times B$) across four fundamental computing models: **Sequential CPU Baseline**, **OpenMP Shared-Memory Multi-Threading**, **Open MPI Distributed-Memory Clustering**, and **CUDA GPU Massively Parallel Acceleration**.

Each paradigm executes an identical workload of 128 GFLOPs ($16,000,000$ output elements) with deterministic mathematical verification ($C[i][j] = 4000.00$).

---

## 2. Key Findings

- **CUDA achieves massive speedup**: 0.34 s ($711.66\times$ total speedup, $770.40\times$ kernel speedup across 16,000,000 GPU threads).
- **OpenMP is fastest CPU model**: 40.55 s (6.02× speedup using 8 threads in shared memory).
- **Open MPI scales across nodes**: 92.98 s (2.63× speedup across a 4-node VM cluster).
- **Sequential baseline is slowest**: 244.12 s (single-core CPU execution).
- **Hierarchy of Parallelism**: GPU Acceleration (0.34 s) $\gg$ Shared CPU RAM (40.55 s) $\gg$ Multi-Node Network (92.98 s) $\gg$ Single Core (244.12 s).
- **Results verified**: All four implementations produced the exact same output ($C[0][0] = 4000.00$).

---

## 3. Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Key Findings](#2-key-findings)
3. [Table of Contents](#3-table-of-contents)
4. [Problem Definition & Mathematical Model](#4-problem-definition--mathematical-model)
5. [Performance Comparison & Benchmark Matrix](#5-performance-comparison--benchmark-matrix)
6. [Comparison of Parallel Computing Paradigms](#6-comparison-of-parallel-computing-paradigms)
   * [6.1 Sequential CPU Baseline](#61-sequential-cpu-baseline)
   * [6.2 OpenMP Shared-Memory Multi-Threading](#62-openmp-shared-memory-multi-threading)
   * [6.3 Open MPI Multi-Node Distributed Cluster](#63-open-mpi-multi-node-distributed-cluster)
   * [6.4 CUDA GPU Massively Parallel Acceleration](#64-cuda-gpu-massively-parallel-acceleration)
7. [Architectural Deep-Dive & System Trade-Offs](#7-architectural-deep-dive--system-trade-offs)
8. [Cluster Topology & Hardware Specifications](#8-cluster-topology--hardware-specifications)
9. [Step-by-Step Reproduction Guide](#9-step-by-step-reproduction-guide)
10. [Repository Structure](#10-repository-structure)

---

## 4. Problem Definition & Mathematical Model

The experiment computes the dense matrix multiplication $C = A \times B$ where $A, B \in \mathbb{R}^{N \times N}$ and $N = 4000$:

$$C[i][j] = \sum_{k=0}^{3999} A[i][k] \times B[k][j] \quad \text{for } 0 \le i, j < 4000$$

### Computational & Memory Invariants:
* **Output Dimensions**: $4000 \times 4000 = 16,000,000$ distinct output elements.
* **Arithmetic Complexity**: Each cell requires 4000 multiplications and 4000 additions:
  $$\text{Total Floating-Point Operations} = 2 \times N^3 = 2 \times (4000)^3 = 128,000,000,000 \text{ FLOPs } (128 \text{ GFLOPs})$$
* **Storage Footprint**:
  * Double-precision ($8\text{ bytes/element}$): $4000 \times 4000 \times 8\text{ bytes} \approx 128\text{ MB/matrix}$ ($384\text{ MB total}$).
  * Single-precision ($4\text{ bytes/element}$, CUDA): $4000 \times 4000 \times 4\text{ bytes} \approx 64\text{ MB/matrix}$ ($192\text{ MB total}$).
* **Verification Proof**: Because $A[i][k] = 1.0$ and $B[k][j] = 1.0$ for all elements:
  $$C[i][j] = \sum_{k=0}^{3999} (1.0 \times 1.0) = 4000.00$$
  Exact mathematical verification requires $C[0][0] = 4000.00$ and $C[N-1][N-1] = 4000.00$.

---

## 5. Performance Comparison & Benchmark Matrix

| Paradigm | Architecture Model | Compute Resources | Execution Time | Speedup ($S = \frac{T_{seq}}{T_{p}}$) | Parallel Efficiency ($\frac{S}{P}$) | Verification $C[0][0]$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sequential** | Single-threaded CPU | 1 CPU Core | **244.120000 s** | **1.00×** (Baseline) | 100.0% (Ref) | `4000.00` (PASS) |
| **Open MPI** | Distributed cluster | 4 Nodes (VMs) | **92.979510 s** | **2.63×** | **65.8%** | `4000.00` (PASS) |
| **OpenMP** | Shared-memory thread pool | 8 vCPU Cores | **40.545825 s** | **6.02×** | **75.3%** | `4000.00` (PASS) |
| **CUDA** | GPU SIMT Acceleration | 16,000,000 Threads | **0.343028 s** | **711.66×** | — | `4000.00` (PASS) |

*(Note: Total CUDA phase time includes Host-to-Device transfer, kernel execution time of 0.316872 s [770.40× speedup], and Device-to-Host transfer).*

```
Execution Time (Lower is Better - Dramatic Performance Scaling)
Sequential Baseline : [████████████████████████████████████████] 244.12 s (1.00x Baseline)
Open MPI (4 VMs)    : [███████████████                        ] 92.98 s  (2.63x speedup)
OpenMP (8 Cores)    : [██████                                ] 40.55 s  (6.02x speedup)
CUDA GPU Phase      : [▏                                     ] 0.34 s   (711.66x speedup)
```

---

## 6. Comparison of Parallel Computing Paradigms

### 6.1 Sequential CPU Baseline
* **Model**: Single execution thread executing standard $O(N^3)$ loop in row-major order.
* **Mechanism**: Contiguous heap allocations (`malloc`) avoid stack overflow for 128 MB structures.
* **Analysis**: While Matrix $A$ benefits from unit-stride cache hits, Matrix $B$ causes a cache miss almost every iteration as memory jumps by 32 KB per column lookup.
* **Detailed Technical Report**: [View Sequential Experiment](./Sequential.md)

### 6.2 OpenMP Shared-Memory Multi-Threading
* **Model**: Fork-join multi-threading under a symmetric multiprocessing (SMP) architecture.
* **Mechanism**: Loop iterations partitioned via `#pragma omp parallel for private(j, k) schedule(static)` across 8 threads.
* **Concurrency Features**:
  * Private stack scoping for `j` and `k` prevents race conditions.
  * Static chunking partitions rows ($500\text{ rows/thread}$), guaranteeing zero false sharing across cache lines on Matrix $C$.
* **Detailed Technical Report**: [View OpenMP Experiment](./OpenMP.md)

### 6.3 Open MPI Multi-Node Distributed Cluster
* **Model**: Shared-nothing architecture communicating explicitly over a TCP/IP virtual network.
* **Cluster Nodes**:
  * **Master (`master`)**: `192.168.190.128` (Rank 0)
  * **Worker 1 (`worker1`)**: `192.168.190.129` (Rank 1)
  * **Worker 2 (`worker2`)**: `192.168.190.130` (Rank 2)
  * **Worker 3 (`worker3`)**: `192.168.190.131` (Rank 3)
* **Collective Operations**:
  * `MPI_Scatter`: Partitions Matrix $A$ into four 1000-row chunks ($32\text{ MB}$ each) from Rank 0 to all workers.
  * `MPI_Bcast`: Broadcasts the entire Matrix $B$ ($128\text{ MB}$) to all ranks.
  * `MPI_Gather`: Gathers partial results back to Rank 0 into global Matrix $C$.
* **Detailed Technical Report**: [View Open MPI Experiment](./MPI.md)

### 6.4 CUDA GPU Massively Parallel Acceleration
* **Model**: SIMT (Single Instruction, Multiple Threads) architecture running on NVIDIA GPU hardware.
* **Execution Grid**:
  * Block dimensions: $16 \times 16$ threads (256 threads per block).
  * Grid dimensions: $250 \times 250$ blocks (62,500 thread blocks).
  * Total concurrent threads: **16,000,000 logical threads** (1 dedicated thread per matrix element).
* **Data Flow**: Host allocates pinned/heap memory, transfers $A$ and $B$ to device memory via PCIe (`cudaMemcpyHostToDevice`), invokes the GPU kernel, and transfers $C$ back (`cudaMemcpyDeviceToHost`).
* **Detailed Technical Report**: [View CUDA Experiment](./CUDA.md)

---

## 7. Architectural Deep-Dive & System Trade-Offs

### Cross-Paradigm Architectural Comparison

| Dimension | OpenMP Shared Memory | Open MPI Distributed Memory | CUDA GPU Accelerator |
| :--- | :--- | :--- | :--- |
| **Hardware Target** | Multi-core CPU socket | Multi-node VM / bare-metal cluster | NVIDIA GPU Streaming Multiprocessors |
| **Concurrency Scale** | 8 threads | 4 independent nodes (processes) | 16,000,000 threads (62,500 blocks) |
| **Address Space** | Unified virtual address space | Disjoint private memory per node | Dedicated high-bandwidth VRAM (GDDR/HBM) |
| **Data Access Latency** | Nanosecond-scale RAM/L3 cache | Millisecond-scale TCP/IP Ethernet packet transfer | Terabyte/s on-chip memory bandwidth |
| **Data Movement** | Zero-copy shared read of Matrix B | Explicit serialization of 128 MB broadcast | Explicit DMA transfer over PCIe bus |
| **Scaling Horizon** | Motherboard socket limits | Scalable to thousands of cluster nodes | Scalable across multi-GPU / NVLink topologies |

---

## 8. Cluster Topology & Hardware Specifications

```text
               [ Master Node ]
            IP: 192.168.190.128 (Rank 0)
                   │  │  │
    ┌──────────────┼──┼──┴─────────────┐
    ▼                 ▼                ▼
[ Worker 1 ]     [ Worker 2 ]     [ Worker 3 ]
192.168.190.129  192.168.190.130  192.168.190.131
  (Rank 1)         (Rank 2)         (Rank 3)
```

* **Operating System**: Ubuntu 22.04 LTS (WSL2 & VMware Workstation) / Windows CUDA Host
* **CPU Compiler**: GCC 11.4.0 with `-O2` optimization
* **GPU Compiler**: NVIDIA CUDA Compiler Driver (`nvcc`) with `-O2`
* **MPI Implementation**: Open MPI 5.0.10 / 4.1.6 (`mpicc`, `mpirun`)
* **Security & Transport**: OpenSSH Server 8.9p1 with RSA 3072-bit passwordless authentication

---

## 9. Step-by-Step Reproduction Guide

### 1. Compile and Run Sequential:
```bash
gcc -O2 matrix_sequential.c -o matrix_sequential
./matrix_sequential
```

### 2. Compile and Run OpenMP:
```bash
export OMP_NUM_THREADS=8
gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp
./matrix_openmp
```

### 3. Compile and Launch Distributed MPI Cluster:
```bash
# Compile on Master
mpicc -O2 matrix_mpi.c -o matrix_mpi

# Copy binary to worker nodes
scp matrix_mpi worker1:~/matrix_mpi
scp matrix_mpi worker2:~/matrix_mpi
scp matrix_mpi worker3:~/matrix_mpi

# Launch across 4 nodes
mpirun -np 4 --hostfile hosts sh -c '$HOME/matrix_mpi'
```

### 4. Compile and Run CUDA GPU Acceleration:
```bash
# Compile CUDA C++ program
nvcc -O2 matrix_cuda.cu -o matrix_cuda.exe

# Execute CUDA matrix multiplication
./matrix_cuda.exe
```

---

## 10. Repository Structure

```text
Parallel-and-GPU-Computing/
│
├── README.md         # Executive summary, comparative benchmarks & cross-paradigm analysis
├── Sequential.md     # Sequential CPU baseline report, C code & execution output
├── OpenMP.md         # OpenMP multi-threaded report, C code & execution output
├── MPI.md            # Open MPI cluster deployment, 21 setup screenshots & code
├── CUDA.md           # CUDA GPU massively parallel report, .cu code & execution output
├── images/           # All authentic terminal screenshots & output logs
│   ├── sequential_olp.png
│   ├── openmp_olp.png
│   ├── cuda_olp.jpeg
│   └── 01_ping_connectivity.jpeg ... 21_mpi_send_recv_output.jpeg
└── .gitignore        # Ignores compiled binaries and temporary submission files
```
