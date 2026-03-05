#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>

const double PI = 3.14159265358979323846;
const double a = -4.0;
const double b = 4.0;
const int nsteps = 40000000;

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}


double func(double x)
{
    return exp(-x * x);
}


double integrate(double (*func)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;

    for (int i = 0; i < n; i++)
        sum += func(a + h * (i + 0.5));

    sum *= h;

    return sum;
}


double integrate_omp(double (*func)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;

#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; ++i)
        sum += func(a + h * (i + 0.5));

    sum *= h;

    return sum;
}


double integrate_omp_atomic(double (*func)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;

#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
        double fx = func(a + h * (i + 0.5));
#pragma omp atomic
        sum += fx;
    }

    sum *= h;

    return sum;
}


double run_serial()
{
    double t = cpuSecond();
    double res = integrate(func, a, b, nsteps);
    t = cpuSecond() - t;
    ///printf("Result (serial): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}


double run_parallel()
{
    double t = cpuSecond();
    double res = integrate_omp(func, a, b, nsteps);
    t = cpuSecond() - t;
    ///printf("Result (parallel): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}


double run_atomic()
{
    double t = cpuSecond();
    double res = integrate_omp_atomic(func, a, b, nsteps);
    t = cpuSecond() - t;
    ///printf("Result (atomic): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}


void create_speedup_plot_integral(const std::vector<int> &threads,
                                  const std::vector<double> &speedups_serial,
                                  const std::vector<double> &speedups_parallel,
                                  const std::vector<double> &speedups_atomic)
{
    if (threads.empty() ||
        speedups_serial.size() != threads.size() ||
        speedups_parallel.size() != threads.size() ||
        speedups_atomic.size() != threads.size())
        return;

    std::ostringstream name_suffix;
    name_suffix << nsteps;

    std::string data_filename = "speedup_integral_" + name_suffix.str() + ".dat";
    std::string script_filename = "speedup_integral_" + name_suffix.str() + ".plt";
    std::string image_filename = "speedup_integral_" + name_suffix.str() + ".png";

    std::ofstream data_file(data_filename.c_str());
    if (!data_file)
    {
        std::cerr << "Cannot open file " << data_filename << " for writing" << std::endl;
        return;
    }
    for (std::size_t i = 0; i < threads.size(); ++i)
    {
        data_file << threads[i] << " "
                  << speedups_serial[i] << " "
                  << speedups_parallel[i] << " "
                  << speedups_atomic[i] << "\n";
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
        << "set title 'Speedup vs threads (integral, nsteps=" << nsteps << ")'\n"
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
    script_file << "plot '" << data_filename << "' using 1:2 with linespoints title 'serial', \\\n"
                << "     '' using 1:3 with linespoints title 'parallel', \\\n"
                << "     '' using 1:4 with linespoints title 'atomic'\n";
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


void run_experiment()
{
    const int threads_list[] = {1, 2, 4, 8, 16, 20, 40};

    std::cout << std::fixed << std::setprecision(6);

    double T_serial = run_serial();

    std::cout << "T_serial = " << T_serial << " sec" << std::endl;
    std::cout << std::endl;

    std::cout << std::setw(8) << "n"
              << std::setw(15) << "T_par"
              << std::setw(15) << "S_par"
              << std::setw(15) << "T_at"
              << std::setw(15) << "S_at" << std::endl;

    std::vector<int> used_threads;
    std::vector<double> speedups_serial;
    std::vector<double> speedups_parallel;
    std::vector<double> speedups_atomic;

    for (std::size_t i = 0; i < sizeof(threads_list) / sizeof(threads_list[0]); ++i)
    {
        int nthreads = threads_list[i];

        omp_set_num_threads(nthreads);
        double T_par = run_parallel();

        omp_set_num_threads(nthreads);
        double T_at = run_atomic();

        double S_par = (T_par > 0.0) ? (T_serial / T_par) : 0.0;
        double S_at = (T_at > 0.0) ? (T_serial / T_at) : 0.0;

        std::cout << std::setw(8) << nthreads
                  << std::setw(15) << T_par
                  << std::setw(15) << S_par
                  << std::setw(15) << T_at
                  << std::setw(15) << S_at << std::endl;

        used_threads.push_back(nthreads);
        speedups_serial.push_back(1.0);
        speedups_parallel.push_back(S_par);
        speedups_atomic.push_back(S_at);
    }

    create_speedup_plot_integral(used_threads, speedups_serial, speedups_parallel, speedups_atomic);
}


void print_system_info() {
    std::cout << std::endl << "=== CPU (lscpu) ===" << std::endl;
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


int main(int argc, char **argv)
{

    print_system_info();

    std::cout << std::endl << "=== Integral speedup experiments ===" << std::endl;
    
    run_experiment();

    return 0;
}