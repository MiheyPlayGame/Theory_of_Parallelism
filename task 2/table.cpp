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
#include <omp.h>

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

/*
 * matrix_vector_product: Compute matrix-vector product c[m] = a[m][n] * b[n]
 */
void matrix_vector_product(double *a, double *b, double *c, size_t m, size_t n)
{
    for (int i = 0; i < m; i++)
    {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

/*
    matrix_vector_product_omp: Compute matrix-vector product c[m] = a[m][n] * b[n]
*/
void matrix_vector_product_omp(double *a, double *b, double *c, size_t m, size_t n)
{
#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++)
        {
            c[i] = 0.0;
            for (int j = 0; j < n; j++)
                c[i] += a[i * n + j] * b[j];
        }
    }
}

double run_serial(size_t n, size_t m)
{
    double *a, *b, *c;
    a = (double*)malloc(sizeof(*a) * m * n);
    b = (double*)malloc(sizeof(*b) * n);
    c = (double*)malloc(sizeof(*c) * m);

    if (a == NULL || b == NULL || c == NULL)
    {
        free(a);
        free(b);
        free(c);
        std::cerr << "Error allocate memory in run_serial for size "
                  << "M=" << m << ", N=" << n << std::endl;
        return -1.0;
    }

    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
            a[i * n + j] = i + j;
    }

    for (size_t j = 0; j < n; j++)
        b[j] = j;

    double t = cpuSecond();
    matrix_vector_product(a, b, c, m, n);
    t = cpuSecond() - t;

    free(a);
    free(b);
    free(c);
    return t;
}

double run_parallel(size_t n, size_t m, int nthreads)
{
    double *a, *b, *c;

    a = (double*)malloc(sizeof(*a) * m * n);
    b = (double*)malloc(sizeof(*b) * n);
    c = (double*)malloc(sizeof(*c) * m);

    if (a == NULL || b == NULL || c == NULL)
    {
        free(a);
        free(b);
        free(c);
        std::cerr << "Error allocate memory in run_parallel for size "
                  << "M=" << m << ", N=" << n
                  << ", threads=" << nthreads << std::endl;
        return -1.0;
    }

    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
            a[i * n + j] = i + j;
    }

    for (size_t j = 0; j < n; j++)
        b[j] = j;

    omp_set_num_threads(nthreads);
    double t = cpuSecond();
    matrix_vector_product_omp(a, b, c, m, n);
    t = cpuSecond() - t;

    free(a);
    free(b);
    free(c);
    return t;
}

void create_speedup_plot(size_t size, const std::vector<int> &threads, const std::vector<double> &speedups)
{
    if (threads.empty() || speedups.empty() || threads.size() != speedups.size())
        return;

    std::ostringstream name_suffix;
    name_suffix << size;

    std::string data_filename = "speedup_DGEMV_" + name_suffix.str() + ".dat";
    std::string script_filename = "speedup_DGEMV_" + name_suffix.str() + ".plt";
    std::string image_filename = "speedup_DGEMV_" + name_suffix.str() + ".png";

    std::ofstream data_file(data_filename.c_str());
    if (!data_file)
    {
        std::cerr << "Cannot open file " << data_filename << " for writing" << std::endl;
        return;
    }
    for (std::size_t i = 0; i < threads.size(); ++i)
    {
        data_file << threads[i] << " " << speedups[i] << "\n";
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
        << "set title 'Speedup vs threads (M=N=" << size << ")'\n"
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
    script_file << "plot '" << data_filename << "' using 1:2 with linespoints title 'S_n'\n";
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

    std::cout << std::fixed << std::setprecision(6);

    for (size_t sz_idx = 0; sz_idx < 2; ++sz_idx)
    {
        size_t M = sizes[sz_idx];
        size_t N = sizes[sz_idx];

        std::cout << std::endl;
        std::cout << "Matrix size M = N = " << M << std::endl;

        double T_serial = run_serial(N, M);
        if (T_serial < 0.0)
        {
            std::cout << "Skip size " << M << " (not enough memory)" << std::endl;
            continue;
        }

        std::cout << "T_serial = " << T_serial << " sec" << std::endl;
        std::cout << std::endl;

        std::cout << std::setw(8) << "n"
                  << std::setw(15) << "T_n (sec)"
                  << std::setw(15) << "S_n=T1/Tn" << std::endl;

        std::vector<int> used_threads;
        std::vector<double> used_speedups;

        for (size_t i = 0; i < sizeof(threads_list) / sizeof(threads_list[0]); ++i)
        {
            int nthreads = threads_list[i];

            double T_n = run_parallel(N, M, nthreads);
            if (T_n < 0.0)
            {
                std::cout << std::setw(8) << nthreads
                          << std::setw(15) << "N/A"
                          << std::setw(15) << "N/A" << std::endl;
                continue;
            }
            double S_n = (T_n > 0.0) ? (T_serial / T_n) : 0.0;

            std::cout << std::setw(8) << nthreads
                      << std::setw(15) << T_n
                      << std::setw(15) << S_n << std::endl;

            used_threads.push_back(nthreads);
            used_speedups.push_back(S_n);
        }

        create_speedup_plot(M, used_threads, used_speedups);
    }
}

void print_system_info() {
    std::cout << "=== CPU (lscpu) ===" << std::endl;
    std::cout << std::endl;
    std::system("lscpu");

    std::cout << std::endl << "=== Product name (cat /sys/devices/virtual/dmi/id/product_name) ===" << std::endl;
    std::cout << std::endl;
    std::system("cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo \"(no product_name in this environment)\"");

    std::cout << std::endl << "=== NUMA nodes (numactl --hardware) ===" << std::endl;
    std::cout << std::endl;
    std::system("numactl --hardware 2>/dev/null || echo \"(numactl not available or no NUMA info)\"");

    std::cout << std::endl << "=== Memory per node (from numactl/free) ===" << std::endl;
    std::cout << std::endl;
    std::system("free -h || echo \"(free not available)\"");

    std::cout << std::endl << "=== OS (cat /etc/os-release) ===" << std::endl;
    std::cout << std::endl;
    std::system("cat /etc/os-release");
}

int main(int argc, char *argv[])
{
    // First, print system information
    print_system_info();

    std::cout << std::endl << "=== DGEMV tests ===" << std::endl;

    // Run experiments for requested sizes and thread counts
    run_experiments();

    return 0;
}

