#include "gwo.hpp"
using Point = Eigen::Array2d;
static std::vector<Point> dataSet;
std::vector<Point> getDataSet(int n)
{
    std::vector<Point> dataSet;
    float dx = 1.0f / n;
    for (int i = 0; i <= n; i++)
    {
        float x = i * dx;
        float y = std::sin(2 * M_PI * x);
        y += GWO::random(-0.07f, 0.07f);
        dataSet.push_back({x, y});
    }
    return dataSet;
}
struct ExampleProblem : public GWO::Wolf<double>
{
    double polyEval(float input) const
    {
        double sum = 0;
        for (int i = 0; i < GWO::constants::N; i++)
        {
            sum += std::pow(input, i) * pos[i];
        }
        return sum;
    }
    double fitness() const override
    {
        double err = 0;
        for (auto &point : dataSet)
        {
            double diff = polyEval(point[0]) - point[1];
            err += diff * diff;
        }
        return err / dataSet.size();
    }
};
int main()
{
    dataSet = getDataSet(9);
    auto wolf = GWO::run<ExampleProblem>(500);
    std::cout << wolf.savedFitness << "\n";
    std::cout << wolf << "\n";

    return 0;
}