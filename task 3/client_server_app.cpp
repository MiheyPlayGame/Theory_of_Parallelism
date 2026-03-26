#include "task_server.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

struct ClientSpec
{
    std::string name;
    std::function<std::pair<double, double>(std::mt19937 &)> arg_generator;
    std::function<double(double, double)> op;
    std::uint32_t seed = 0;
};

static void write_client_file_header(std::ofstream &out, const std::string &op_name, std::size_t n)
{
    out << "# client_op: " << op_name << "\n";
    out << "# tasks: " << n << "\n";
}

// Три клиента по условию задания:
// 1) синус, аргумент случайный;
// 2) квадратный корень, аргумент случайный;
// 3) степень, основание и показатель случайные.

static ClientSpec make_assignment_client_sin()
{
    return ClientSpec{
        "sin",
        [](std::mt19937 &rng) {
            std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
            return std::make_pair(dist(rng), 0.0);
        },
        [](double a, double) { return std::sin(a); },
        0xA11CE001u};
}

static ClientSpec make_assignment_client_sqrt()
{
    return ClientSpec{
        "sqrt",
        [](std::mt19937 &rng) {
            std::uniform_real_distribution<double> dist(0.0, 1.0e12);
            return std::make_pair(dist(rng), 0.0);
        },
        [](double a, double) { return std::sqrt(a); },
        0xA11CE002u};
}

static ClientSpec make_assignment_client_pow()
{
    return ClientSpec{
        "pow",
        [](std::mt19937 &rng) {
            std::uniform_real_distribution<double> base_dist(0.0, 1000.0);
            std::uniform_real_distribution<double> exp_dist(-10.0, 10.0);
            return std::make_pair(base_dist(rng), exp_dist(rng));
        },
        [](double a, double b) { return std::pow(a, b); },
        0xA11CE003u};
}

// Ровно три клиента из задания, дополнительные типы можно добавить в другой вектор и join'ить с этим.
static std::vector<ClientSpec> assignment_clients()
{
    return {
        make_assignment_client_sin(),
        make_assignment_client_sqrt(),
        make_assignment_client_pow(),
    };
}

static void run_client(TaskServer<double> &srv, std::size_t N, const fs::path &out_dir, const ClientSpec &spec)
{
    std::mt19937 rng(spec.seed);
    const fs::path out_path = out_dir / ("client_" + spec.name + ".txt");

    std::ofstream out(out_path);
    if (!out)
        throw std::runtime_error("cannot open output file: " + out_path.string());

    write_client_file_header(out, spec.name, N);
    out << std::setprecision(17);

    struct SubmittedTask
    {
        std::size_t id = 0;
        double a = 0.0;
        double b = 0.0;
        double expected = 0.0;
    };

    std::vector<SubmittedTask> submitted;
    submitted.reserve(N);

    for (std::size_t i = 0; i < N; ++i)
    {
        const auto [a, b] = spec.arg_generator(rng);
        const auto id = srv.add_task([op = spec.op, a, b]() { return op(a, b); });
        submitted.push_back(SubmittedTask{id, a, b, spec.op(a, b)});
    }

    for (const SubmittedTask &task : submitted)
    {
        const double result = srv.request_result(task.id);
        out << task.id << " " << task.a << " " << task.b << " " << task.expected << " " << result << "\n";
        srv.erase_result(task.id);
    }
}

int main(int argc, char **argv)
{
    std::size_t N = 5000;
    std::size_t workers = std::max<std::size_t>(1, std::thread::hardware_concurrency());

    if (argc >= 2)
        N = static_cast<std::size_t>(std::stoull(argv[1]));
    if (argc >= 3)
        workers = static_cast<std::size_t>(std::stoull(argv[2]));

    if (N <= 5 || N >= 10000)
    {
        std::cerr << "N must satisfy 5 < N < 10000. Got N=" << N << "\n";
        return 2;
    }

    const fs::path out_dir = fs::path("task 3") / "results";
    fs::create_directories(out_dir);

    TaskServer<double> server(workers); // Thread Pool (worker threads)
    server.start();

    const std::vector<ClientSpec> clients = assignment_clients();
    std::vector<std::thread> client_threads;
    client_threads.reserve(clients.size());

    std::exception_ptr eptr = nullptr;
    std::mutex eptr_mu;

    for (const ClientSpec &spec_ref : clients)
    {
        const ClientSpec spec = spec_ref;
        client_threads.emplace_back([&server, N, &out_dir, spec, &eptr, &eptr_mu] {
            try
            {
                run_client(server, N, out_dir, spec);
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lk(eptr_mu);
                if (!eptr)
                    eptr = std::current_exception();
            }
        });
    }

    for (auto &th : client_threads)
    {
        if (th.joinable())
        {
            th.join();
        }
    }

    server.stop();

    if (eptr)
    {
        try
        {
            std::rethrow_exception(eptr);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Client error: " << e.what() << "\n";
        }
        return 1;
    }

    std::cout << "Wrote results to: " << (out_dir.string()) << "\n";
    return 0;
}

