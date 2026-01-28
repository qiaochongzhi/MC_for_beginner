#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "MonteCarlo.h"

namespace py = pybind11;

// Helper struct to expose GCAResult
struct PyGCAResult {
    double d_pot;
    double d_vir;
    int clusterSize;
};

PYBIND11_MODULE(MonteCarlo, m)
{
    // Register GCAResult struct
    py::class_<PyGCAResult>(m, "GCAResult")
        .def(py::init<>())
        .def_readwrite("d_pot", &PyGCAResult::d_pot)
        .def_readwrite("d_vir", &PyGCAResult::d_vir)
        .def_readwrite("clusterSize", &PyGCAResult::clusterSize);
    
    py::class_<MonteCarlo>(m, "MonteCarlo")
      .def(py::init<int, int, double, double, bool, bool>())
      .def("SetPosition", &MonteCarlo::SetPosition)
      .def("GetPosition", &MonteCarlo::GetPosition)
      .def("GetTemperature", &MonteCarlo::GetTemperature)
      .def("GetPotential", &MonteCarlo::GetPotential)
      .def("MoveParticles", &MonteCarlo::MoveParticles)
      .def("TestParticles", &MonteCarlo::testParticles)
      .def("InitPosition", &MonteCarlo::InitPosition)
      .def("SetFileName", &MonteCarlo::setFileName)
      .def("SetInitStep", &MonteCarlo::SetInitStep)
      .def("SetMethod", &MonteCarlo::SetMethod, "Set simulation method: 'Metropolis' or 'GCA'")
      .def("SetVerbose", &MonteCarlo::SetVerbose, "Set verbose mode for output control")
      .def("MCrun", &MonteCarlo::NVTrun)
      // .def("SetBox", &MonteCarlo::SetBox);
      .def("SetBox", static_cast<void (MonteCarlo::*)(const py::array_t<float> &)>(&MonteCarlo::SetBox))
      // GCA debugging methods
      .def("StepGCA", [](MonteCarlo &self) {
          auto result = self.StepGCA();
          PyGCAResult pyResult;
          pyResult.d_pot = result.d_pot;
          pyResult.d_vir = result.d_vir;
          pyResult.clusterSize = result.clusterSize;
          return pyResult;
      }, "Perform one GCA step and return result")
      .def("GetLJDiff", [](MonteCarlo &self,
                           const std::vector<double>& r_old,
                           const std::vector<double>& r_new,
                           const std::vector<double>& r_j) {
          double d_pot, d_vir;
          self.GetLJDiff(r_old, r_new, r_j, d_pot, d_vir);
          return py::make_tuple(d_pot, d_vir);
      }, "Calculate LJ difference between positions")
      .def("DebugPrintPositions", [](MonteCarlo &self) {
          auto positions = self.GetPosition();
          auto view = positions.unchecked<2>();
          for (py::ssize_t i = 0; i < view.shape(0); i++) {
              std::cout << "Particle " << i << ": ";
              for (py::ssize_t j = 0; j < view.shape(1); j++) {
                  std::cout << view(i, j) << " ";
              }
              std::cout << std::endl;
          }
      }, "Print all particle positions for debugging");
}