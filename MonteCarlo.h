#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class MonteCarlo
{
    public:
    MonteCarlo( int numberOfParticles, int dim, double t, double rCut );
    void SetBox( const py::array_t<float>& box );
    void SetPosition( const py::array_t<float>& position );
    py::array_t<float> GetPosition();
    double GetTemperature() const { return temperature; }
    py::object GetPotential();
    py::object displacementParticles( double drMax );
    double testParticles();

    private:
    int numberOfParticles;
    int dim = 3;
    double sr2Over = 1.77; // Overlap threshold ( pot > 100 )
    double rCut    = 2.5;
    double temperature;

    std::vector<double> box = {10., 10., 10.};
    std::vector< std::vector<double> > position;

    int calculateInteraction( const std::vector<double>& p1, const std::vector<double>& p2, double& pot, double& vir );
    int calculateTotalPotential( double& pot, double& vir );

    int metropolis( double delta );
};