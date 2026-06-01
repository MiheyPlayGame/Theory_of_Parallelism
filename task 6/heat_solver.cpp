/**
 * Task 6: 2D heat conduction (stationary Laplace), 5-point stencil, Jacobi iteration.
 * Method: simple iteration x_{n+1} = x_n - tau*(A*x_n - b), equivalent to Jacobi
 * with u_new[i,j] = (u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1]) / 4 on interior nodes.
 *
 * Boundaries: linear interpolation between corners (10, 20, 30, 20).
 * Build with NVIDIA HPC SDK: pgc++ -acc -Minfo=all ...
 */

#include <boost/program_options.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace po = boost::program_options;

using clock_type = std::chrono::high_resolution_clock;

static double elapsed_sec(const clock_type::time_point &t0) {
    const auto dt = clock_type::now() - t0;
    return std::chrono::duration<double>(dt).count();
}

static void set_boundary(double *u, int n) {
    const double c00 = 10.0;  // (0, 0)
    const double c0N = 20.0;  // (0, n-1)
    const double cNN = 30.0;  // (n-1, n-1)
    const double cN0 = 20.0;  // (n-1, 0)
    const double inv = 1.0 / static_cast<double>(n - 1);

    u[0] = c00;
    u[n - 1] = c0N;
    u[(n - 1) * n + (n - 1)] = cNN;
    u[(n - 1) * n] = cN0;

    for (int j = 1; j < n - 1; ++j) {
        const double t = static_cast<double>(j) * inv;
        u[j] = c00 + (c0N - c00) * t;                         // top
        u[(n - 1) * n + j] = cN0 + (cNN - cN0) * t;           // bottom
    }
    for (int i = 1; i < n - 1; ++i) {
        const double t = static_cast<double>(i) * inv;
        u[i * n] = c00 + (cN0 - c00) * t;                     // left
        u[i * n + (n - 1)] = c0N + (cNN - c0N) * t;           // right
    }
}

static void init_grid(double *u, int n) {
    const std::size_t sz = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    std::fill(u, u + sz, 0.0);
    set_boundary(u, n);
}

static void print_grid(const double *u, int n) {
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (j) std::cout << ' ';
            std::cout << std::setw(8) << u[i * n + j];
        }
        std::cout << '\n';
    }
    std::cout << std::defaultfloat << std::setprecision(6);
}

#if defined(HEAT_BASELINE)
#define HEAT_VARIANT_NAME "baseline"
#else
#define HEAT_VARIANT_NAME "optimized"
#endif

// ---------------------------------------------------------------------------
// Baseline: per-iteration host/device sync via update + separate reduction.
// ---------------------------------------------------------------------------
#if defined(HEAT_BASELINE)

static int solve_jacobi(double *u, double *v, int n, int max_iter, double eps,
                        int &out_iters, double &out_err) {
    const int n2 = n * n;
    out_iters = max_iter;
    out_err = std::numeric_limits<double>::infinity();

#pragma acc data copy(u[0:n2], v[0:n2])
    {
        double *cur = u;
        double *nxt = v;

        for (int iter = 0; iter < max_iter; ++iter) {
#pragma acc parallel loop collapse(2) present(cur[0:n2], nxt[0:n2])
            for (int i = 1; i < n - 1; ++i) {
                for (int j = 1; j < n - 1; ++j) {
                    const int k = i * n + j;
                    nxt[k] = 0.25 * (cur[k - n] + cur[k + n] + cur[k - 1] + cur[k + 1]);
                }
            }

            double err = 0.0;
#pragma acc parallel loop collapse(2) reduction(max : err) present(cur[0:n2], nxt[0:n2])
            for (int i = 1; i < n - 1; ++i) {
                for (int j = 1; j < n - 1; ++j) {
                    const int k = i * n + j;
                    const double d = std::fabs(nxt[k] - cur[k]);
                    if (d > err) err = d;
                }
            }

            std::swap(cur, nxt);
            out_err = err;
            out_iters = iter + 1;
            if (err < eps) break;
        }

        if (cur != u) {
            std::memcpy(u, cur, static_cast<std::size_t>(n2) * sizeof(double));
        }
    }
    return 0;
}

#else
// ---------------------------------------------------------------------------
// Optimized: persistent device buffers, fused update+error, pointer swap only.
// ---------------------------------------------------------------------------

static int solve_jacobi(double *u, double *v, int n, int max_iter, double eps,
                        int &out_iters, double &out_err) {
    const int n2 = n * n;
    out_iters = max_iter;
    out_err = std::numeric_limits<double>::infinity();

#pragma acc data create(u[0:n2], v[0:n2])
    {
#pragma acc update device(u[0:n2], v[0:n2])

        double *cur = u;
        double *nxt = v;

        for (int iter = 0; iter < max_iter; ++iter) {
            double err = 0.0;

#pragma acc parallel loop collapse(2) vector_length(256) \
    present(cur[0:n2], nxt[0:n2]) reduction(max : err)
            for (int i = 1; i < n - 1; ++i) {
                for (int j = 1; j < n - 1; ++j) {
                    const int k = i * n + j;
                    const double newv =
                        0.25 * (cur[k - n] + cur[k + n] + cur[k - 1] + cur[k + 1]);
                    const double d = std::fabs(newv - cur[k]);
                    nxt[k] = newv;
                    if (d > err) err = d;
                }
            }

            std::swap(cur, nxt);
            out_err = err;
            out_iters = iter + 1;
            if (err < eps) break;
        }

        if (cur != u) {
            std::memcpy(u, cur, static_cast<std::size_t>(n2) * sizeof(double));
        }
#pragma acc update self(u[0:n2])
    }
    return 0;
}

#endif

int main(int argc, char **argv) {
    int n = 128;
    double eps = 1e-6;
    int max_iter = 1000000;
    bool print_matrix = false;
    bool quiet = false;

    po::options_description desc("2D heat equation (5-point Jacobi), OpenACC");
    desc.add_options()("help,h", "print help")(
        "size,n", po::value<int>(&n)->default_value(128),
        "grid size N (NxN), use 128/256/512/1024 for benchmarks")(
        "eps,e", po::value<double>(&eps)->default_value(1e-6), "tolerance")(
        "max-iter,m", po::value<int>(&max_iter)->default_value(1000000),
        "maximum iterations")(
        "print-grid,p", po::bool_switch(&print_matrix),
        "print full grid after solve")(
        "quiet,q", po::bool_switch(&quiet),
        "only print iterations and error (for scripts)");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }

    if (vm.count("help")) {
        std::cout << desc << '\n';
        std::cout << "Variant compiled: " << HEAT_VARIANT_NAME << '\n';
        return 0;
    }

    if (n < 3) {
        std::cerr << "Grid size must be >= 3\n";
        return 1;
    }

    const int n2 = n * n;
    std::vector<double> a(static_cast<std::size_t>(n2));
    std::vector<double> b(static_cast<std::size_t>(n2));

    init_grid(a.data(), n);
    init_grid(b.data(), n);

    const auto t0 = clock_type::now();
    int iters = 0;
    double err = 0.0;
    solve_jacobi(a.data(), b.data(), n, max_iter, eps, iters, err);
    const double elapsed = elapsed_sec(t0);

    if (!quiet) {
        std::cout << "variant=" << HEAT_VARIANT_NAME << " N=" << n << '\n';
    }
    std::cout << "iterations=" << iters << " error=" << std::scientific << err
              << std::defaultfloat << " time_sec=" << std::fixed << std::setprecision(6)
              << elapsed << '\n';

    if (print_matrix || n == 10 || n == 13) {
        std::cout << "\n--- grid " << n << "x" << n << " ---\n";
        print_grid(a.data(), n);
    }

    return 0;
}
