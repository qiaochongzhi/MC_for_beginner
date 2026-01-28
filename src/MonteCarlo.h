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
#include <fstream>

#include <random>
#include <string>

#include "linkList.h"
#include "PotentialType.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Define the simulation method
enum class SimulationMethod
{
    METROPOLIS,
    GCA
};

class MonteCarlo
{
    public:
    MonteCarlo( int numberOfParticles, int dim, double t, double rCut, bool isNeighbourList, bool verbose = false );
    void SetBox( const py::array_t<float>& box );
    void SetBox( const std::vector<double>& box );
    void SetPosition( const py::array_t<float>& position );
    void SetInitStep( const size_t n );
    void InitPosition();
    py::array_t<float> GetPosition();
    double GetTemperature() const { return temperature; }
    py::object GetPotential();
    py::object MoveParticles( const double drMax );
    std::map<std::string, double> displacementParticles( double drMax );
    double testParticles();

    void setFileName(std::string);

    // Set simulation method
    void SetMethod(std::string method);

    // Set verbose mode
    void SetVerbose(bool v);

    //py::array_t<float> NVTrun( int nStep, double drMax );
    std::map<std::string, std::vector<double>> NVTrun( int nStep, double drMax, int interval = 100 );

    std::string filename = "trajectory.xyz";

    // ======= GCA ======== //
    struct GCAResult
    {
        double d_pot;
        double d_vir;
        int clusterSize;
    };

    GCAResult StepGCA();
    void GetLJDiff(const std::vector<double>& r_old, const std::vector<double>& r_new, const std::vector<double>& r_j, double& d_pot, double& d_vir);
    // ==================== //

    private:
    int numberOfParticles;
    int dim = 3;
    double sr2Over = 1.77; // Overlap threshold ( pot > 100 )
    double rCut    = 2.5;
    double temperature;

    size_t nInit = 1000;

    std::mt19937 gen;
    std::uniform_real_distribution<double> dist;

    double RandNumber();

    bool isNeighbourList;
    bool verbose = false;
    linkList neighbourList;

    std::vector<double> box = {10., 10., 10.};
    std::vector< std::vector<double> > position;

    SimulationMethod m_Method = SimulationMethod::METROPOLIS;

    // ======= GCA ======== //

    std::vector<bool> m_InCluster;
    std::vector<bool> m_IsCandidate;
    std::vector<int> m_Stack;
    std::vector<int> m_CandidateList;

    std::vector<double> m_BoundaryPotAcc;
    std::vector<double> m_BoundaryVirAcc;

    // ==================== //

    PotentialType calculateInteraction( const std::vector<double>& p1, const std::vector<double>& p2 );
    PotentialType calculateTotalPotential();

    bool metropolis( double delta );
    void writeXYZTrajectory(const std::string& filename, int timestep, bool append = true);

};

#endif // MONTECARLO_H