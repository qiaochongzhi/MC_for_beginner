#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "MonteCarlo.h"

namespace py = pybind11;

PYBIND11_MODULE(MonteCarlo, m)
{
    py::class_<MonteCarlo>(m, "MonteCarlo")
      .def(py::init<int, int, double, double, bool>())
      .def("SetPosition", &MonteCarlo::SetPosition)
      .def("GetPosition", &MonteCarlo::GetPosition)
      .def("GetTemperature", &MonteCarlo::GetTemperature)
      .def("GetPotential", &MonteCarlo::GetPotential)
      .def("MoveParticles", &MonteCarlo::MoveParticles)
      .def("TestParticles", &MonteCarlo::testParticles)
      .def("InitPosition", &MonteCarlo::InitPosition)
      .def("MCrun", &MonteCarlo::NVTrun)
      // .def("SetBox", &MonteCarlo::SetBox);
      .def("SetBox", static_cast<void (MonteCarlo::*)(const py::array_t<float> &)>(&MonteCarlo::SetBox));
}