#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>


template <class T>
class TaskServer
{
public:
    using id_type = std::size_t;
    using task_type = std::function<T()>;

    explicit TaskServer(std::size_t worker_count = 1)
        : worker_count_(worker_count == 0 ? 1 : worker_count)
    {
    }

    TaskServer(const TaskServer &) = delete;
    TaskServer &operator=(const TaskServer &) = delete;

    ~TaskServer()
    {
        stop();
    }

    void start()
    {
        bool expected = false;
        if (!running_.compare_exchange_strong(expected, true))
            return;

        stopping_.store(false);
        workers_.clear();
        workers_.reserve(worker_count_);
        for (std::size_t i = 0; i < worker_count_; ++i)
            workers_.emplace_back([this]() { worker_loop(); });
    }

    void stop()
    {
        if (!running_.load())
            return;

        stopping_.store(true);
        {
            std::lock_guard<std::mutex> lk(mu_);
        }
        cv_.notify_all();

        for (auto &t : workers_)
        {
            if (t.joinable())
                t.join();
        }
        workers_.clear();
        running_.store(false);
    }

    template <class F>
    id_type add_task(F &&f)
    {
        if (!running_.load())
            throw std::runtime_error("TaskServer::add_task(): server not started");

        task_type fn = task_type(std::forward<F>(f));

        const id_type id = next_id_.fetch_add(1, std::memory_order_relaxed);

        std::packaged_task<T()> task([fn = std::move(fn)]() mutable -> T { return fn(); });
        std::shared_future<T> fut = task.get_future().share();

        {
            std::lock_guard<std::mutex> lk(mu_);
            tasks_.emplace_back(id, std::move(task));
            results_.emplace(id, std::move(fut));
        }
        cv_.notify_one();
        return id;
    }

    T request_result(id_type id)
    {
        std::shared_future<T> fut;
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = results_.find(id);
            if (it == results_.end())
                throw std::out_of_range("TaskServer::request_result(): unknown id");
            fut = it->second;
        }
        return fut.get();
    }

    std::optional<T> try_request_result(id_type id)
    {
        std::shared_future<T> fut;
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = results_.find(id);
            if (it == results_.end())
                return std::nullopt;
            fut = it->second;
        }

        if (fut.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
            return std::nullopt;
        return fut.get();
    }
    bool erase_result(id_type id)
    {
        std::lock_guard<std::mutex> lk(mu_);
        return results_.erase(id) != 0;
    }

    std::size_t results_size() const
    {
        std::lock_guard<std::mutex> lk(mu_);
        return results_.size();
    }

private:
    using queued_task = std::pair<id_type, std::packaged_task<T()>>;

    void worker_loop()
    {
        for (;;)
        {
            queued_task item;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this]() { return stopping_.load() || !tasks_.empty(); });

                if (tasks_.empty())
                {
                    if (stopping_.load())
                        return;
                    continue;
                }

                item = std::move(tasks_.front());
                tasks_.pop_front();
            }

            try
            {
                item.second();
            }
            catch (...)
            {
                // Exception is already captured by packaged_task future state.
            }
        }
    }

    const std::size_t worker_count_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<queued_task> tasks_;
    std::unordered_map<id_type, std::shared_future<T>> results_;

    std::vector<std::thread> workers_;
    std::atomic<id_type> next_id_{1};
    std::atomic<bool> running_{false};
    std::atomic<bool> stopping_{false};
};

