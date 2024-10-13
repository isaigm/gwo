#include "gwo.hpp"
using Point = Eigen::Array2d;

struct Problem : public GWO::Problem<double>
{
    Problem(GWO::Setup setup) : GWO::Problem<double>(setup)
    {
        int n = 9;
        float dx = 1.0f / n;
        for (int i = 0; i <= n; i++)
        {
            float x = i * dx;
            float y = std::sin(2 * M_PI * x);
            y += GWO::random(-0.07f, 0.07f);
            dataSet.push_back({x, y});
        }
    }
    double polyEval(float input, const Eigen::ArrayX<double> &pos) const
    {
        double sum = 0;
        for (size_t i = 0; i < setup.N; i++)
        {
            sum += std::pow(input, i) * pos[i];
        }
        return sum;
    }
    double fitness(const Eigen::ArrayX<double> &pos) const override
    {
        double err = 0;
        for (auto &point : dataSet)
        {
            double diff = polyEval(point[0], pos) - point[1];
            err += diff * diff;
        }
        return err / dataSet.size();
    }
    std::vector<Point> dataSet;
};
int main()
{
    GWO::Setup setup{.N = 10, .POP_SIZE = 100, .maxRange = 15.0, .minRange = -15.0};
    Problem problem(setup);
    auto wolf = problem.run(500);
    std::cout << wolf.savedFitness << "\n";
    std::cout << wolf << "\n";
    return 0;
}