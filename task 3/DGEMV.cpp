#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <thread>
#include <algorithm>
#include <exception>

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

static std::size_t normalize_thread_count(int nthreads)
{
    if (nthreads <= 0)
        return 1;
    return static_cast<std::size_t>(nthreads);
}

template <class Fn>
static void parallel_for(std::size_t begin, std::size_t end, std::size_t nthreads, Fn fn)
{
    if (end <= begin)
        return;

    nthreads = std::min<std::size_t>(nthreads, end - begin);
    if (nthreads <= 1)
    {
        fn(begin, end);
        return;
    }

    const std::size_t total = end - begin;
    const std::size_t base = total / nthreads;
    const std::size_t rem = total % nthreads;

    std::vector<std::thread> threads;
    threads.reserve(nthreads);

    std::size_t cur = begin;
    for (std::size_t t = 0; t < nthreads; ++t)
    {
        const std::size_t chunk = base + (t < rem ? 1 : 0);
        const std::size_t lb = cur;
        const std::size_t ub = cur + chunk;
        cur = ub;
        threads.emplace_back([=, &fn]() { fn(lb, ub); });
    }

    for (auto &th : threads)
        th.join();
}

static void init_matrix_vector_parallel(std::vector<double> &a,
                                       std::vector<double> &b,
                                       std::size_t m,
                                       std::size_t n,
                                       std::size_t nthreads)
{
    parallel_for(0, m, nthreads, [&](std::size_t i0, std::size_t i1) {
        for (std::size_t i = i0; i < i1; ++i)
        {
            const std::size_t row = i * n;
            for (std::size_t j = 0; j < n; ++j)
                a[row + j] = static_cast<double>(i + j);
        }
    });

    parallel_for(0, n, nthreads, [&](std::size_t j0, std::size_t j1) {
        for (std::size_t j = j0; j < j1; ++j)
            b[j] = static_cast<double>(j);
    });
}

static void dgemv_threads(const std::vector<double> &a,
                          const std::vector<double> &b,
                          std::vector<double> &c,
                          std::size_t m,
                          std::size_t n,
                          std::size_t nthreads)
{
    parallel_for(0, m, nthreads, [&](std::size_t i0, std::size_t i1) {
        for (std::size_t i = i0; i < i1; ++i)
        {
            const std::size_t row = i * n;
            double sum = 0.0;
            for (std::size_t j = 0; j < n; ++j)
                sum += a[row + j] * b[j];
            c[i] = sum;
        }
    });
}

double run_threads(size_t n, size_t m, int nthreads)
{
    const std::size_t nt = normalize_thread_count(nthreads);
    try
    {
        std::vector<double> a(m * n);
        std::vector<double> b(n);
        std::vector<double> c(m);

        init_matrix_vector_parallel(a, b, m, n, nt);

        double t = cpuSecond();
        dgemv_threads(a, b, c, m, n, nt);
        t = cpuSecond() - t;
        return t;
    }
    catch (const std::bad_alloc &)
    {
        std::cerr << "Error allocate memory in run_threads for size "
                  << "M=" << m << ", N=" << n
                  << ", threads=" << nthreads << std::endl;
        return -1.0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error in run_threads: " << e.what() << std::endl;
        return -1.0;
    }
}

void create_combined_speedup_plot(const std::vector<size_t> &sizes,
                                  const std::vector<int> &threads,
                                  const std::vector<std::vector<double>> &speedups_by_size)
{
    if (threads.empty() || sizes.empty() || sizes.size() != speedups_by_size.size())
        return;

    for (std::size_t k = 0; k < speedups_by_size.size(); ++k)
    {
        if (speedups_by_size[k].size() != threads.size())
            return;
    }

    std::string data_filename = "speedup_DGEMV_combined.dat";
    std::string script_filename = "speedup_DGEMV_combined.plt";
    std::string image_filename = "speedup_DGEMV_combined.png";

    std::ofstream data_file(data_filename.c_str());
    if (!data_file)
    {
        std::cerr << "Cannot open file " << data_filename << " for writing" << std::endl;
        return;
    }
    for (std::size_t i = 0; i < threads.size(); ++i)
    {
        data_file << threads[i];
        for (std::size_t k = 0; k < speedups_by_size.size(); ++k)
            data_file << " " << speedups_by_size[k][i];
        data_file << "\n";
    }
    data_file.close();

    std::ofstream script_file(script_filename.c_str());
    if (!script_file)
    {
        std::cerr << "Cannot open file " << script_filename << " for writing" << std::endl;
        return;
    }
    script_file
        << "set terminal pngcairo size 800,600\n"
        << "set output '" << image_filename << "'\n"
        << "set title 'Speedup vs threads for DGEMV'\n"
        << "set xlabel 'Number of threads'\n"
        << "set ylabel 'Speedup S_n = T_serial / T_n'\n"
        << "set grid\n"
        << "set xtics (";
    for (std::size_t i = 0; i < threads.size(); ++i)
    {
        script_file << "'" << threads[i] << "' " << threads[i];
        if (i + 1 < threads.size())
            script_file << ", ";
    }
    script_file << ")\n";
    script_file << "plot ";
    for (std::size_t k = 0; k < sizes.size(); ++k)
    {
        if (k != 0)
            script_file << ", ";
        script_file << "'" << data_filename << "' using 1:" << (k + 2)
                    << " with linespoints title 'M=N=" << sizes[k] << "'";
    }
    script_file << "\n";
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
        // Remove intermediate files, leave only PNG
        std::remove(data_filename.c_str());
        std::remove(script_filename.c_str());
    }
}

void run_experiments()
{
    const size_t sizes[] = {20000, 30000};
    const int threads_list[] = {1, 2, 4, 8, 16, 20, 40};
    const std::size_t sizes_count = sizeof(sizes) / sizeof(sizes[0]);
    const std::size_t threads_count = sizeof(threads_list) / sizeof(threads_list[0]);

    std::cout << std::fixed << std::setprecision(6);
    std::vector<std::vector<double>> all_speedups(sizes_count, std::vector<double>(threads_count, 0.0));
    std::vector<bool> size_completed(sizes_count, false);

    for (size_t sz_idx = 0; sz_idx < sizes_count; ++sz_idx)
    {
        size_t M = sizes[sz_idx];
        size_t N = sizes[sz_idx];

        std::cout << std::endl;
        std::cout << "Matrix size M = N = " << M << std::endl;

        double T1 = run_threads(N, M, 1);
        if (T1 < 0.0)
        {
            std::cout << "Skip size " << M << " (not enough memory)" << std::endl;
            continue;
        }

        std::cout << "T_1 = " << T1 << " sec" << std::endl;
        std::cout << std::endl;

        std::cout << std::setw(8) << "n"
                  << std::setw(15) << "T_n (sec)"
                  << std::setw(15) << "S_n=T1/Tn" << std::endl;

        for (size_t i = 0; i < threads_count; ++i)
        {
            int nthreads = threads_list[i];

            double T_n = run_threads(N, M, nthreads);
            if (T_n < 0.0)
            {
                std::cout << std::setw(8) << nthreads
                          << std::setw(15) << "N/A"
                          << std::setw(15) << "N/A" << std::endl;
                continue;
            }
            double S_n = (T_n > 0.0) ? (T1 / T_n) : 0.0;

            std::cout << std::setw(8) << nthreads
                      << std::setw(15) << T_n
                      << std::setw(15) << S_n << std::endl;

            all_speedups[sz_idx][i] = S_n;
        }
        size_completed[sz_idx] = true;
    }

    std::vector<size_t> plotted_sizes;
    std::vector<std::vector<double>> plotted_speedups;
    for (std::size_t i = 0; i < sizes_count; ++i)
    {
        if (!size_completed[i])
            continue;
        plotted_sizes.push_back(sizes[i]);
        plotted_speedups.push_back(all_speedups[i]);
    }

    if (!plotted_sizes.empty())
    {
        std::vector<int> threads_vec(threads_list, threads_list + threads_count);
        create_combined_speedup_plot(plotted_sizes, threads_vec, plotted_speedups);
    }
}

int main(int argc, char *argv[])
{
    std::cout << std::endl << "=== DGEMV tests ===" << std::endl;

    // Run experiments for requested sizes and thread counts
    run_experiments();

    return 0;
}

