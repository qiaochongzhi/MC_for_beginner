#include <vector>
#include <iostream>
#include <math.h>
#include <stdlib.h>

#include "MonteCarlo.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#define RandNumber() ( rand() / double( RAND_MAX ) )
#define INFINT 1.e10

namespace py = pybind11;

MonteCarlo::MonteCarlo(int num, int dim, double t, double rCut)
{
    this->numberOfParticles = num;
    this->dim               = dim;
    this->temperature       = t;
    this->rCut              = rCut;

    position.resize(num);

    for ( int i = 0; i < num; i++ )
        position[i].resize(dim);
}

void MonteCarlo::SetBox( const py::array_t<float>& box )
{
    auto r = box.unchecked<1>();
    for ( int i = 0; i < dim; i++ )
        this->box[i] = r(i);

}

void MonteCarlo::SetPosition(const py::array_t<float>& position)
{
    auto r = position.unchecked<2>();
    for ( py::size_t i = 0; i < r.shape(0); i++)
        for ( py::size_t j = 0; j < r.shape(1); j++)
            this->position[i][j] = r(i, j);

    // Test the position
    /*
    for ( int i = 0; i < numberOfParticles; i++ )
    {
        for ( int j = 0; j < dim; j++ )
        {
            std::cout << this->position[i][j] << " ";
        }
        std::cout << std::endl;
    }
    */

    /*
    double pot;
    double vir;
    double rCut = 2.5;

    int Flag = calculateInteraction(this->position[0], this->position[1], rCut, pot, vir);
    std::cout << "pot = " << pot << std::endl;
    std::cout << "vir = " << vir << std::endl;
    std::cout << "Flag = " << Flag << std::endl;
    */
}

py::array_t<float> MonteCarlo::GetPosition()
{
    int shape[2]{numberOfParticles, dim};
    auto r    = py::array_t<float>(shape);
    auto view = r.mutable_unchecked<2>();

    for (int i = 0; i < numberOfParticles; i++)
        for ( int j = 0; j < dim; j++ )
            view(i, j) = this->position[i][j];

    return r;
}

int MonteCarlo::calculateInteraction(const std::vector<double>& p1, const std::vector<double>& p2, double& pot, double& vir)
{
    int overlap  = 0;
    double r     = 0.0;
    double dr    = 0.0;
    double rCut2 = rCut * rCut;

    double sr2, sr6, sr12;

    pot = 0.0;
    vir = 0.0;
    for ( int i = 0; i < dim; i++ )
    {
        dr = p1[i] - p2[i];
        dr = dr - round( dr ); // Periodic boundary condition.
        dr = dr * box[i];      // Set the dr as real distance.
        r  = r + dr * dr;
    }

    sr2 = 1. / r;
    if ( sr2 > sr2Over )
    {
        overlap = 1;
        return overlap;
    }

    if ( r < rCut2 )
    {
        sr6  = pow( sr2, 3 );
        sr12 = sr6 * sr6;

        pot  = sr12 - sr6;
        vir  = pot + sr12;

        pot  = pot * 4.0;
        vir  = vir * 24.0 / 3.0;
    }

    return overlap;
}

int MonteCarlo::calculateTotalPotential( double& pot, double& vir )
{
    double pot1 = 0.0;
    double vir1 = 0.0;
    int overlap = 0;

    pot = 0.0;
    vir = 0.0;
    for ( int i = 0; i < numberOfParticles; i++ )
    {
        for ( int j = i + 1; j < numberOfParticles; j++ )
        {
            overlap = calculateInteraction( position[i], position[j], pot1, vir1 );
            if ( overlap )
                return overlap;

            pot = pot + pot1;
            vir = vir + vir1;
        }
    }
    return overlap;
}

py::object MonteCarlo::GetPotential()
{
    double pot = 0.0;
    double vir = 0.0;
    int overlap;
    auto potential = py::dict();

    overlap = calculateTotalPotential( pot, vir );

    potential["pot"] = pot;
    potential["vir"] = vir;
    potential["overlap"] = overlap;

    //std::cout << "pot = " << pot << std::endl;
    //std::cout << "vir = " << vir << std::endl;

    return potential;
}

py::object MonteCarlo::displacementParticles( double drMax )
{
    auto result = py::dict();
    int move = 0;

    std::vector<double> pos(3, 0);
    std::vector<double> poso(3, 0);

    double totalPotential = 0.0;
    double totalVirial    = 0.0;

    double delta;

    int moves = 0;
    for ( int i = 0; i < numberOfParticles; i++ )
    {
        int overlap   = 0;
        int overlap0  = 0;

        double potT  = 0.0;
        double virT  = 0.0;

        double potT0 = 0.0;
        double virT0 = 0.0;
        for ( int j = 0; j < dim; j++ )
        {
            pos[j]  = position[i][j] + ( 2*RandNumber() - 1 ) * drMax;
            poso[j] = position[i][j];

            pos[j] = pos[j] - round( pos[j] ); // PBC
        }

        double vir  = 0.0;
        double vir0 = 0.0;
        double pot  = 0.0;
        double pot0 = 0.0;
        for ( int k = 0; k < numberOfParticles; k++ )
        {
            if ( i == k ) continue;
            overlap  = calculateInteraction( pos,  position[k], pot,  vir );
            overlap0 = calculateInteraction( poso, position[k], pot0, vir0 );

            if ( overlap0 )
            {
                std::cout << "Error: overlap, displacement particles." << std::endl;
                exit(-1);
            }

            if ( overlap )
                break;
            else
            {
                potT  = potT  + pot;
                potT0 = potT0 + pot0;

                virT  = virT  + vir;
                virT0 = virT0 + vir0;
            }
        }

        if ( !overlap )
        {
            delta = potT - potT0;
            delta = delta / temperature;

            if ( metropolis( delta ) )
            {
                for ( int j = 0; j < dim; j++ )
                    position[i][j] = pos[j];
                totalVirial    += ( virT - virT0 );
                totalPotential += ( potT - potT0 );
                moves += 1;
            }
        }
    }

    result["pot"]   = totalPotential;
    result["vir"]   = totalVirial;
    result["moves"] = moves;

    return result;
}

bool MonteCarlo::metropolis( double delta )
{
    double trial = 0;
    double exponent_guard = 75.0;

    if ( delta > exponent_guard )
        return false;
    else if ( delta < 0.0 )
        return true;

    trial = RandNumber();
    if ( exp( -delta ) > trial )
        return true;
    else
        return false;
}

double MonteCarlo::testParticles()
{
    double overlap  = 0;
    double potTotal = 0.0;
    double vir      = 0.0;
    double pot      = 0.0;

    std::vector<double> pos(3,0);
    for ( int j = 0; j < dim; j++ )
        pos[j] = RandNumber() - 0.5;

    for ( int i = 0; i < numberOfParticles; i++)
    {
        overlap = calculateInteraction( pos, position[i], pot, vir);
        if ( overlap )
            return INFINT;

        potTotal = potTotal + pot;
    }

    return potTotal;
}
