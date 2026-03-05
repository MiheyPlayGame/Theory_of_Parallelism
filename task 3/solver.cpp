#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <omp.h>
#include <stdio.h>
#include <time.h>

#define MAX_ITER 1000
#define TOL 1e-10

// ---------- helper: high‑resolution timer (same as in integral.cpp) ----------
double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

// ---------- serial Jacobi (baseline) ----------
int solve_jacobi_serial(const double *a, const double *b, double *x, size_t n,
                        int max_iter, double tol)
{
    if (n == 0 || a == nullptr || b == nullptr || x == nullptr)
        return -1;
    for (size_t i = 0; i < n; i++)
        if (std::abs(a[i * n + i]) < 1e-15)
            return -1;

    // Richardson iteration parameter; chosen conservatively based on n
    double tau = 1.0 / (2.0 * static_cast<double>(n));

    std::vector<double> x_old(n, 0.0);
    int iter;

    for (iter = 0; iter < max_iter; iter++)
    {
        // Richardson update based on residual r = b - A x_old
        double res_norm = 0.0;
        for (size_t i = 0; i < n; i++)
        {
            double sum = 0.0;
            for (size_t j = 0; j < n; j++)
            {
                sum += a[i * n + j] * x_old[j];
            }
            double ri = b[i] - sum;
            x[i] = x_old[i] + tau * ri;
            res_norm += ri * ri;
        }

        res_norm = std::sqrt(res_norm);
        if (res_norm <= tol)
            return iter + 1;

        std::memcpy(x_old.data(), x, n * sizeof(double));
    }
    return -1; // not converged
}

// ---------- parallel Jacobi (blocks version, already correct) ----------
int solve_jacobi_parallel_blocks(const double *a, const double *b, double *x, size_t n,
                                 int max_iter, double tol)
{
    if (n == 0 || a == nullptr || b == nullptr || x == nullptr)
        return -1;
    for (size_t i = 0; i < n; i++)
        if (std::abs(a[i * n + i]) < 1e-15)
            return -1;

    // Richardson iteration parameter; chosen conservatively based on n
    double tau = 1.0 / (2.0 * static_cast<double>(n));

    std::vector<double> x_old(n, 0.0);
    int iter;

    for (iter = 0; iter < max_iter; iter++)
    {
        double res_norm = 0.0;

#pragma omp parallel for
        for (size_t i = 0; i < n; i++)
        {
            double sum = 0.0;
            for (size_t j = 0; j < n; j++)
            {
                sum += a[i * n + j] * x_old[j];
            }
            double ri = b[i] - sum;
            x[i] = x_old[i] + tau * ri;
        }

#pragma omp parallel for reduction(+ : res_norm)
        for (size_t i = 0; i < n; i++)
        {
            double ri = b[i];
            for (size_t j = 0; j < n; j++)
                ri -= a[i * n + j] * x_old[j];
            res_norm += ri * ri;
        }
        res_norm = std::sqrt(res_norm);
        if (res_norm <= tol)
            return iter + 1;

        std::memcpy(x_old.data(), x, n * sizeof(double));
    }
    return -1;
}

// Keeps threads alive across iterations to reduce fork‑join overhead.
int solve_jacobi_parallel_whole(const double *a, const double *b, double *x, size_t n,
    int max_iter, double tol)
{
    if (n == 0 || a == nullptr || b == nullptr || x == nullptr)
        return -1;
    for (size_t i = 0; i < n; i++)
        if (std::abs(a[i * n + i]) < 1e-15)
            return -1;

    // Richardson iteration parameter; chosen conservatively based on n
    double tau = 1.0 / (2.0 * static_cast<double>(n));

    std::vector<double> x_old(n, 0.0);
    int converged = 0;
    int last_iter = -1;
    double res_norm = 0.0;
    double local_res = 0.0;

    #pragma omp parallel shared(x_old, x, converged, last_iter, res_norm, local_res)
    {
        for (int iter = 0; iter < max_iter && !converged; iter++)
        {
            // Richardson update – worksharing across threads
            #pragma omp for
            for (size_t i = 0; i < n; i++)
            {
                double sum = 0.0;
                for (size_t j = 0; j < n; j++)
                {
                    sum += a[i * n + j] * x_old[j];
                }
                double ri = b[i] - sum;
                x[i] = x_old[i] + tau * ri;
            }

            #pragma omp single
            local_res = 0.0;

            #pragma omp for reduction(+:local_res)
            for (size_t i = 0; i < n; i++)
            {
                double ri = b[i];
                for (size_t j = 0; j < n; j++)
                    ri -= a[i * n + j] * x_old[j];
                local_res += ri * ri;
            }

            #pragma omp single
            {
                res_norm = std::sqrt(local_res);
                if (res_norm <= tol)
                {
                    converged = 1;
                    last_iter = iter + 1;
                }
                if (!converged)
                    std::memcpy(x_old.data(), x, n * sizeof(double));
            }
        }
    }

    return converged ? last_iter : -1;
}

// ---------- matrix generation ----------
void generate_diag_dominant_matrix(std::vector<double> &a, std::vector<double> &b,
                                   size_t n, double diag_scale = 1.0)
{
    a.resize(n * n);
    b.resize(n);
    // matrix: A[i][i] = n, A[i][j] = 1.0 for i != j
    for (size_t i = 0; i < n; i++)
    {
        double row_sum = 0.0;
        for (size_t j = 0; j < n; j++)
        {
            if (i == j)
                a[i * n + j] = 2.0;
            else
                a[i * n + j] = 1.0;
            row_sum += a[i * n + j];
        }
        // choose b so that exact solution is all ones
        b[i] = row_sum;
    }
}

// ---------- run a single test and return time (seconds) ----------
double run_jacobi(int (*solver)(const double*, const double*, double*, size_t, int, double),
                  const double *a, const double *b, double *x, size_t n,
                  int max_iter, double tol, int nthreads = 1)
{
    omp_set_num_threads(nthreads);
    double t = cpuSecond();
    int iters = solver(a, b, x, n, max_iter, tol);
    t = cpuSecond() - t;
    if (iters < 0)
        std::cerr << "Warning: solver did not converge (threads=" << nthreads << ")\n";
    return t;
}

// ---------- create speedup plot (similar to integral.cpp) ----------
void create_speedup_plot_jacobi(size_t n,
                                const std::vector<int> &threads,
                                const std::vector<double> &speedups_blocks,
                                const std::vector<double> &speedups_whole)
{
    if (threads.empty() ||
        speedups_blocks.size() != threads.size() ||
        speedups_whole.size() != threads.size())
        return;

    std::ostringstream name_suffix;
    name_suffix << n;

    std::string data_filename = "speedup_solver_" + name_suffix.str() + ".dat";
    std::string script_filename = "speedup_solver_" + name_suffix.str() + ".plt";
    std::string image_filename = "speedup_solver_" + name_suffix.str() + ".png";

    std::ofstream data_file(data_filename.c_str());
    if (!data_file)
    {
        std::cerr << "Cannot open file " << data_filename << " for writing\n";
        return;
    }
    for (std::size_t i = 0; i < threads.size(); ++i)
    {
        data_file << threads[i] << " "
                  << speedups_blocks[i] << " "
                  << speedups_whole[i] << "\n";
    }
    data_file.close();

    std::ofstream script_file(script_filename.c_str());
    if (!script_file)
    {
        std::cerr << "Cannot open file " << script_filename << " for writing\n";
        return;
    }
    script_file
        << "set terminal pngcairo size 800,600\n"
        << "set output '" << image_filename << "'\n"
        << "set title 'Jacobi solver speedup (matrix size " << n << ")'\n"
        << "set xlabel 'Number of threads'\n"
        << "set ylabel 'Speedup S_n = T_{serial} / T_n'\n"
        << "set grid\n"
        << "set xtics (";
    for (std::size_t i = 0; i < threads.size(); ++i)
    {
        script_file << "'" << threads[i] << "' " << threads[i];
        if (i + 1 < threads.size())
            script_file << ", ";
    }
    script_file << ")\n";
    script_file << "plot '" << data_filename << "' using 1:2 with linespoints title 'blocks', \\\n"
                << "     '' using 1:3 with linespoints title 'whole region'\n";
    script_file.close();

    std::string cmd = "gnuplot " + script_filename;
    int rc = std::system(cmd.c_str());
    if (rc != 0)
    {
        std::cerr << "gnuplot command failed with code " << rc
                  << ". Install gnuplot to generate PNG plots.\n";
    }
    else
    {
        std::remove(data_filename.c_str());
        std::remove(script_filename.c_str());
    }
}

// ---------- experiment driver ----------
void run_experiments()
{
    // matrix sizes to test (adjust based on available memory)
    const size_t sizes[] = {15000};
    const int threads_list[] = {1, 2, 4, 8, 16, 20, 40};
    const int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const int n_threads = sizeof(threads_list) / sizeof(threads_list[0]);

    std::cout << std::fixed << std::setprecision(6);

    for (int s = 0; s < n_sizes; ++s)
    {
        size_t n = sizes[s];
        std::cout << "\n========== Matrix size n = " << n << " ==========\n";

        // generate matrix and RHS
        std::vector<double> a, b, x_serial(n, 0.0);
        generate_diag_dominant_matrix(a, b, n, 1.0);

        // serial time
        std::fill(x_serial.begin(), x_serial.end(), 0.0);
        double T_serial = run_jacobi(solve_jacobi_serial, a.data(), b.data(),
                                     x_serial.data(), n, MAX_ITER, TOL, 1);
        std::cout << "T_serial = " << T_serial << " sec\n\n";

        std::cout << std::setw(8) << "threads"
                  << std::setw(15) << "T_blocks"
                  << std::setw(15) << "S_blocks"
                  << std::setw(15) << "T_whole"
                  << std::setw(15) << "S_whole" << "\n";

        std::vector<int> used_threads;
        std::vector<double> speedups_blocks, speedups_whole;

        for (int t = 0; t < n_threads; ++t)
        {
            int nthreads = threads_list[t];

            // blocks version
            std::vector<double> x_blocks(n, 0.0);
            double T_blocks = run_jacobi(solve_jacobi_parallel_blocks,
                                         a.data(), b.data(), x_blocks.data(),
                                         n, MAX_ITER, TOL, nthreads);
            double S_blocks = (T_blocks > 0.0) ? T_serial / T_blocks : 0.0;

            // whole version
            std::vector<double> x_whole(n, 0.0);
            double T_whole = run_jacobi(solve_jacobi_parallel_whole,
                                        a.data(), b.data(), x_whole.data(),
                                        n, MAX_ITER, TOL, nthreads);
            double S_whole = (T_whole > 0.0) ? T_serial / T_whole : 0.0;

            std::cout << std::setw(8) << nthreads
                      << std::setw(15) << T_blocks
                      << std::setw(15) << S_blocks
                      << std::setw(15) << T_whole
                      << std::setw(15) << S_whole << "\n";

            used_threads.push_back(nthreads);
            speedups_blocks.push_back(S_blocks);
            speedups_whole.push_back(S_whole);
        }

        create_speedup_plot_jacobi(n, used_threads, speedups_blocks, speedups_whole);
    }
}

// ---------- main ----------
int main()
{
    std::cout << "\n=== Jacobi solver experiments ===\n";
    run_experiments();
    return 0;
}