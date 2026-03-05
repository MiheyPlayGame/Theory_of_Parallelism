# Task 3 — OpenMP Speedup Analysis Conclusions

## System Information

| Parameter | Value |
|-----------|-------|
| **CPU** | AMD Ryzen 9 7940HS w/ Radeon 780M Graphics |
| **Physical cores / Logical CPUs** | 8 cores / 16 threads (SMT) |
| **L1d / L1i** | 256 KiB × 8 instances each |
| **L2** | 8 MiB × 8 instances |
| **L3** | 16 MiB (shared) |
| **RAM** | 7.4 GiB total |
| **OS** | Ubuntu 22.04.5 LTS (WSL2 / Microsoft Hyper-V) |

Thread counts tested: **1, 2, 4, 8, 16, 20, 40**.  
Speedup is defined as S\_n = T\_serial / T\_n.

---

## Task 1 — DGEMV (Matrix-Vector Product)

**Source:** `table.cpp`

Row-parallel matrix-vector product **c = A · b** using `#pragma omp parallel` with manual row-range partitioning. Tested for two matrix sizes.

### N = 20 000

![Speedup DGEMV N=20000](task%203/speedup_DGEMV_20000.png)

Working set ≈ 3.2 GB — far exceeds the 16 MiB L3 cache.

### N = 30 000

![Speedup DGEMV N=30000](task%203/speedup_DGEMV_30000.png)

Working set ≈ 7.2 GB.

**Conclusion:** DGEMV is **memory-bandwidth bound**. With both matrix sizes well beyond L3 capacity, threads compete for DRAM bandwidth rather than compute. Speedup grows up to ~4–6× in the physical-core range (≤ 8 threads), then flattens — hyperthreads share the same memory bus and add no extra bandwidth. Beyond 16 logical CPUs, oversubscription inside WSL2 introduces scheduling noise and may cause degradation.

---

## Task 2 — Numerical Integration of exp(–x²)

**Source:** `integral.cpp`  
Integration of `exp(–x²)` over [–4, 4] with **40 000 000** steps. Three variants compared:

| Variant | Directive |
|---------|-----------|
| Serial | — |
| Parallel | `#pragma omp parallel for reduction(+:sum)` |
| Atomic | `#pragma omp parallel for` + `#pragma omp atomic` |

![Speedup Integral N=40000000](task%203/speedup_integral_40000000.png)

**Conclusion:**
- The **`reduction` variant** is nearly embarrassingly parallel — each thread maintains a private partial sum with a single merge at the end. Speedup scales close to linearly up to 8 physical cores and continues to grow moderately with hyperthreads.
- The **`atomic` variant** serialises every individual accumulation through a shared variable. This creates heavy contention among threads, making it slower than the `reduction` approach and, at high thread counts, potentially slower than the serial baseline.
- This experiment clearly demonstrates why `reduction` is the correct OpenMP pattern for parallel accumulation — `atomic` destroys parallel efficiency for high-throughput loops.

---

## Task 3 — Richardson Iterative Solver

**Source:** `solver.cpp`  
Richardson iteration for an N × N diagonally-dominant linear system, N = **15 000**, up to 1000 iterations, tolerance 10⁻¹⁰. Working set ≈ 1.8 GB.

Two parallel variants are compared against the serial baseline:

| Variant | Strategy |
|---------|----------|
| **Blocks** (`solve_jacobi_parallel_blocks`) | `#pragma omp parallel for` is entered and exited on every iteration — fork-join overhead paid each step |
| **Whole** (`solve_jacobi_parallel_whole`) | A single `#pragma omp parallel` wraps the entire iteration loop; threads stay alive across iterations, synchronised with `#pragma omp for` + `#pragma omp single` barriers |

![Speedup Solver N=15000](task%203/speedup_solver_15000.png)

**Conclusion:** Both variants show virtually identical speedup curves across all thread counts. The **whole-region** approach was designed to eliminate repeated fork-join overhead, but in practice the OpenMP runtime on this system already reuses a persistent thread pool — the per-iteration fork-join cost of the blocks variant is negligible compared to the heavy N² compute and memory traffic. Since the dominant bottleneck is DRAM bandwidth (1.8 GB working set), not thread management, both strategies converge to the same performance. Speedup grows up to ~4–8× within the physical-core range (≤ 8 threads) and then flattens due to bandwidth saturation.

---

## Overall Summary

**Key takeaways:**
1. **Memory-bandwidth-bound kernels** (DGEMV, Jacobi) saturate DRAM bandwidth early — going beyond 8 physical cores yields marginal benefit.
2. **Compute-bound kernels** with private accumulators (`reduction`) scale well and benefit from hyperthreading.
3. **Naive synchronisation** (`atomic` on a hot variable) can make parallel code slower than serial; always prefer `reduction` for accumulation.
4. **WSL2 / Hyper-V** introduces measurement noise at high thread counts (> 16), where the hypervisor scheduler becomes an additional bottleneck.
