#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "gwo.hpp"

namespace py = pybind11;

class PyProblem : public GWO::Problem<double> {
public:
    using GWO::Problem<double>::Problem;

    double fitness(const Eigen::ArrayX<double> &pos) const override {
        PYBIND11_OVERRIDE(double, GWO::Problem<double>, fitness, pos);
    }
    
    Eigen::ArrayX<double> fitness_batch(const Eigen::ArrayXX<double>& population_pos) const override {
        PYBIND11_OVERRIDE(
            Eigen::ArrayX<double>,   // Tipo de retorno
            GWO::Problem<double>,    // Clase base
            fitness_batch,           // Nombre del método
            population_pos           // Argumentos
        );
    }
};

PYBIND11_MODULE(gwo_py, m) {
    m.doc() = "Vectorized Python bindings for the GWO algorithm";

    py::class_<GWO::Setup>(m, "Setup")
        .def(py::init<>())
        .def_readwrite("N", &GWO::Setup::N)
        .def_readwrite("POP_SIZE", &GWO::Setup::POP_SIZE)
        .def_readwrite("maxRange", &GWO::Setup::maxRange)
        .def_readwrite("minRange", &GWO::Setup::minRange);

    py::class_<GWO::Wolf<double>>(m, "Wolf")
        .def_readonly("savedFitness", &GWO::Wolf<double>::savedFitness)
        .def_readonly("pos", &GWO::Wolf<double>::pos)
        .def("__repr__", [](const GWO::Wolf<double> &wolf) {
            std::stringstream ss;
            ss << wolf;
            return ss.str();
        });

    py::class_<GWO::Problem<double>, PyProblem>(m, "Problem")
        .def(py::init<GWO::Setup>())
        .def("run", &GWO::Problem<double>::run, "Runs the GWO algorithm", py::arg("maxIterations"))
        .def("getBestKWolves", &GWO::Problem<double>::getBestKWolves, "Get the K best wolves")
        .def_property_readonly("population", [](GWO::Problem<double>& p) { return p.population; })
        // --- CAMBIO 4: EXPONEMOS EL NUEVO MÉTODO A PYTHON ---
        .def("fitness_batch", &GWO::Problem<double>::fitness_batch, "Calculates fitness for the entire population", py::arg("pos_matrix"));
}