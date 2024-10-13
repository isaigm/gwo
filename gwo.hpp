#include <iostream>
#include <random>
#include <vector>
#include <concepts>
#include <queue>
#include <eigen3/Eigen/Dense>
#ifndef GWO_HPP
#define GWO_HPP
namespace GWO
{
    std::random_device rd;
    std::mt19937 mt(rd());
    namespace constants
    {
        size_t K = 3; // best k wolves
    }
    struct Setup
    {
        size_t N{};
        size_t POP_SIZE{};
        float maxRange{};
        float minRange{};
    };
    template <std::floating_point T>
    T random(T min, T max)
    {
        std::uniform_real_distribution<T> distribution(min, max);
        return distribution(mt);
    }
    template <std::floating_point T>
    struct Wolf
    {
        Wolf(size_t n) : pos(n), len(n) {}
        void randomize(T min, T max)
        {
            for (size_t i = 0; i < len; i++)
            {
                pos[i] = random(min, max);
            }
        }
        T savedFitness{};
        Eigen::ArrayX<T> pos;
        size_t len{};
    };
    template <std::floating_point T>
    std::ostream &operator<<(std::ostream &os, const Wolf<T> &wolf)
    {
        std::cout << "[";
        for (size_t i = 0; i < wolf.len - 1; i++)
        {
            std::cout << wolf.pos[i];
            std::cout << ",";
        }
        std::cout << wolf.pos[wolf.len - 1];
        std::cout << "]";
        return os;
    }
    template <std::floating_point T>
    class Comparator
    {
    public:
        bool operator()(const Wolf<T> &w1, const Wolf<T> &w2)
        {
            return w1.savedFitness < w2.savedFitness;
        }
    };
    template <std::floating_point T>
    struct Problem
    {
        virtual T fitness(const Eigen::ArrayX<T> &pos) const = 0;
        Problem(Setup _setup) : nextPos(_setup.N),
                                A(_setup.N), C(_setup.N), setup(_setup)
        {
            for (size_t i = 0; i < setup.POP_SIZE; i++)
            {
                population.emplace_back(setup.N);
                population.back().randomize(setup.minRange, setup.maxRange);
            }
        }
        Wolf<T> run(int maxIterations)
        {
            for (auto &wolf : population)
            {
                addWolf(wolf);
            }
            for (int i = 0; i < maxIterations; i++)
            {
                T a = 2 * (1 - T(i) / T(maxIterations));
                updatePopulation(a);
            }
            return getBestKWolves()[0];
        }
        void addWolf(Wolf<T> &wolf)
        {
            wolf.savedFitness = fitness(wolf.pos);
            heap.push(wolf);
            if (heap.size() > constants::K)
            {
                heap.pop();
            }
        }
        auto getBestKWolves()
        {
            std::vector<Wolf<T>> bestWolves;
            auto copy = heap;
            while (!copy.empty())
            {
                bestWolves.push_back(copy.top());
                copy.pop();
            }
            return bestWolves;
        }
        void updatePopulation(T a)
        {
            auto bestWolves = getBestKWolves();
            for (auto &wolf : population)
            {
                nextPos.setZero();
                for (size_t j = 0; j < constants::K; j++)
                {
                    for (size_t k = 0; k < setup.N; k++)
                    {
                        A[k] = 2 * a * random(0.0, 1.0) - a;
                        C[k] = 2 * random(0.0, 1.0);
                    }
                    auto &bestWolf = bestWolves[j];
                    auto D = (bestWolf.pos * C - wolf.pos).abs();
                    nextPos += bestWolf.pos - D * A;
                }
                nextPos *= (1.0 / T(constants::K));
                wolf.pos = nextPos.max(setup.minRange).min(setup.maxRange);
                addWolf(wolf);
            }
        }
        std::vector<Wolf<T>> population;
        std::priority_queue<Wolf<T>, std::vector<Wolf<T>>, Comparator<T>> heap;
        Eigen::ArrayX<T> nextPos;
        Eigen::ArrayX<T> A;
        Eigen::ArrayX<T> C;
        const Setup setup;
    };
}
#endif
