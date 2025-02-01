#ifndef MONTECARLO_H
#define MONTECARLO_H

#pragma once

#include <vector>
#include <list>
#include <map>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdlib.h>

#include "linkList.h"
#include "PotentialType.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class MonteCarlo
{
    public:
    MonteCarlo( int numberOfParticles, int dim, double t, double rCut, bool isNeighbourList );
    void SetBox( const py::array_t<float>& box );
    void SetBox( const std::vector<double>& box );
    void SetPosition( const py::array_t<float>& position );
    void InitPosition();
    py::array_t<float> GetPosition();
    double GetTemperature() const { return temperature; }
    py::object GetPotential();
    py::object MoveParticles( const double drMax );
    std::map<std::string, double> displacementParticles( double drMax );
    double testParticles();

    //py::array_t<float> NVTrun( int nStep, double drMax );
    std::map<std::string, std::vector<double>> NVTrun( int nStep, double drMax );

    private:
    int numberOfParticles;
    int dim = 3;
    double sr2Over = 1.77; // Overlap threshold ( pot > 100 )
    double rCut    = 2.5;
    double temperature;

    bool isNeighbourList;
    linkList neighbourList;

    std::vector<double> box = {10., 10., 10.};
    std::vector< std::vector<double> > position;

    PotentialType calculateInteraction( const std::vector<double>& p1, const std::vector<double>& p2 );
    PotentialType calculateTotalPotential();

    bool metropolis( double delta );

};

#endif // MONTECARLO_H