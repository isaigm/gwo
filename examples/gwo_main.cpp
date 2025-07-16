#include <iostream>
#include <cmath>
#include "gwo.hpp"

struct SphereProblem : public GWO::Problem<double>
{

    SphereProblem(GWO::Setup setup) : GWO::Problem<double>(setup) {}

    double fitness(const Eigen::ArrayX<double> &pos) const override
    {
        return pos.square().sum();
    }
};
int main()
{

    GWO::Setup setup{
        .N = 5,
        .POP_SIZE = 50,
        .maxRange = (Eigen::ArrayXd(5) << 10.0, 10.0, 10.0, 10.0, 10.0).finished(),
        .minRange = (Eigen::ArrayXd(5) << -10.0, -10.0, -10.0, -10.0, -10.0).finished()};

    SphereProblem problem(setup);

    auto wolf_alfa = problem.run(1000);

    std::cout << "Best fitnes: " << wolf_alfa.savedFitness << "\n";
    std::cout << "Best solution: " << wolf_alfa << "\n";

    return 0;
}