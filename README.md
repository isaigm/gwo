# GWO (Grey Wolf Optimizer)

Implementation of the Grey Wolf Optimizer (GWO) algorithm in C++ with Python bindings.

## Dependencies

- C++20 or above
- Eigen
- pybind11
- Python 3

## Compilation

To compile the Python module, run the following commands from the root of the project:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This will create the Python module in the `build` directory.

## Installation

### System-wide Installation

To install the module system-wide, you can use the following command:

```bash
sudo cmake --install build
```

### Virtual Environment Installation

If you are using a virtual environment like `pipenv`, you can install the module in editable mode using the `setup.py` file:

```bash
pipenv install -e /path/to/gwo
```

Replace `/path/to/gwo` with the absolute path to the project's root directory.

## Usage

The following examples show how to solve the Sphere function benchmark.

### C++ Example (`examples/gwo_main.cpp`)

```cpp
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
```

### Python Example (`gwo_python.py`)

```python
import gwo_py
import numpy as np
import time

class ProblemaEsfera(gwo_py.Problem):
    def __init__(self, setup):
        super().__init__(setup)

    def fitness_batch(self, pos_matrix: np.ndarray) -> np.ndarray:
        return np.sum(pos_matrix**2, axis=1)

config = gwo_py.Setup()
config.N = 5
config.POP_SIZE = 50
config.maxRange = [10.0] * 5
config.minRange = [-10.0] * 5

problema = ProblemaEsfera(config)

start_time = time.time()
lobo_alfa = problema.run(maxIterations=1000)
end_time = time.time()


print(f"Best fitness {lobo_alfa.savedFitness:.6f}")
print(f"Best solution:")
print(lobo_alfa.pos)
print(f"Execution time: {end_time - start_time:.4f} secs")
```
