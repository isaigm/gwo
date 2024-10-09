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
        const size_t N = 10; // number of variables
        const int K = 3;     // best k wolves
        const int POP_SIZE = 100;
        const float maxRange = 15.0f;
        const float minRange = -15.0f;
    }
    template <std::floating_point T>
    T random(T min, T max)
    {
        std::uniform_real_distribution<T> distribution(min, max);
        return distribution(mt);
    }
    template <std::floating_point T>
    struct Wolf
    {
        using Type = T;
        Wolf() : pos(constants::N)
        {
            for (auto i = 0; i < constants::N; i++)
            {
                pos[i] = random(constants::minRange, constants::maxRange);
            }
        }
        T savedFitness {};
        virtual T fitness() const = 0;
        Eigen::ArrayX<T> pos;
    };
    template <std::floating_point T>
    std::ostream &operator<<(std::ostream &os, const Wolf<T> &wolf)
    {
        std::cout << "[";
        for (auto i = 0; i < constants::N; i++)
        {
            std::cout << wolf.pos[i];
            if (i < constants::N - 1)
            {
                std::cout << ",";
            }
        }
        std::cout << "]";
        return os;
    }
    template <typename W>
    concept NumericWolf = (std::floating_point<typename W::Type> && std::derived_from<W, Wolf<typename W::Type>>);

    template <NumericWolf _Wolf>

    class Comparator
    {
    public:
        bool operator()(const _Wolf &w1, const _Wolf &w2)
        {
            return w1.savedFitness < w2.savedFitness;
        }
    };
    template <NumericWolf _Wolf>

    struct GWOState
    {
        using T = _Wolf::Type;
        GWOState() : population(constants::POP_SIZE)
        {
            for (auto &wolf : population)
            {
                addWolf(wolf);
            }
        }
        std::vector<_Wolf> population;
        std::priority_queue<_Wolf, std::vector<_Wolf>, Comparator<_Wolf>> heap;
        void addWolf(_Wolf &wolf)
        {
            wolf.savedFitness = wolf.fitness();
            heap.push(wolf);
            if (heap.size() > constants::K)
            {
                heap.pop();
            }
        }
        auto getBestKWolves()
        {
            std::vector<_Wolf> bestWolves;
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
                Eigen::ArrayX<T> nextPos(constants::N);
                nextPos.setZero();
                for (auto j = 0; j < constants::K; j++)
                {
                    Eigen::ArrayX<T> A(constants::N);
                    Eigen::ArrayX<T> C(constants::N);
                    for (auto k = 0; k < constants::N; k++)
                    {
                        A[k] = 2 * a * random(0.0, 1.0) - a;
                        C[k] = 2 * random(0.0, 1.0);
                    }
                    auto &bestWolf = bestWolves[j];
                    auto D = (bestWolf.pos * C - wolf.pos).abs();
                    nextPos += bestWolf.pos - D * A;
                }
                nextPos *= (1.0 / T(constants::K));
                wolf.pos = nextPos.max(constants::minRange).min(constants::maxRange);
                addWolf(wolf);
            }
        }
    };
    template <NumericWolf _Wolf>
    _Wolf run(int maxIterations)
    {
        using T = _Wolf::Type;

        GWOState<_Wolf> state;
        for (int i = 0; i < maxIterations; i++)
        {
            T a = 2 * (1 - T(i) / T(maxIterations));
            state.updatePopulation(a);
        }
        return state.getBestKWolves()[0];
    }
}
#endif
