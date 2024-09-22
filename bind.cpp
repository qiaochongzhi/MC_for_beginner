#include <pybind11/pybind11.h>
#include "MonteCarlo.h"

namespace py = pybind11;

PYBIND11_MODULE(MonteCarlo, m)
{
    py::class_<MonteCarlo>(m, "MonteCarlo")
      .def(py::init<int, int, double, double>())
      .def("SetPosition", &MonteCarlo::SetPosition)
      .def("GetPosition", &MonteCarlo::GetPosition)
      .def("GetTemperature", &MonteCarlo::GetTemperature)
      .def("GetPotential", &MonteCarlo::GetPotential)
      .def("MoveParticles", &MonteCarlo::displacementParticles)
      .def("TestParticles", &MonteCarlo::testParticles)
      .def("SetBox", &MonteCarlo::SetBox);
}