#include "MonteCarlo.h"

//#define RandNumber() ( rand() / double( RAND_MAX ) )
#define INFINT 1.e10

MonteCarlo::MonteCarlo(int num, int dim, double t, double rCut, bool isNeighbourList, bool verbose)
{
    this->numberOfParticles = num;
    this->dim               = dim;
    this->temperature       = t;
    this->rCut              = rCut;
    this->isNeighbourList   = isNeighbourList;
    this->verbose           = verbose;

    position.resize(num);

    for ( size_t i = 0; i < num; i++ )
        position[i].resize(dim);

    gen = std::mt19937(std::random_device{}()); // init the random seed
    dist = std::uniform_real_distribution<double>(0.0, 1.0);

}

void MonteCarlo::SetVerbose(bool v)
{
    verbose = v;
    neighbourList.setVerbose(v);
}

void MonteCarlo::SetMethod(std::string method)
{
    if (method == "GCA")
    {
        m_Method = SimulationMethod::GCA;

        m_InCluster.resize(numberOfParticles, false);
        m_IsCandidate.resize(numberOfParticles, false);
        m_Stack.reserve(numberOfParticles);
        m_CandidateList.reserve(numberOfParticles);

        m_BoundaryPotAcc.resize(numberOfParticles, 0.0);
        m_BoundaryVirAcc.resize(numberOfParticles, 0.0);

        if (verbose) {
            std::cout << "Method set to: GCA" << std::endl;
        }
    }
    else
    {
        m_Method = SimulationMethod::METROPOLIS;
        if (verbose) {
            std::cout << "Method set to: Metropolis" << std::endl;
        }
    }
}

void MonteCarlo::setFileName(std::string s)
{
    filename = s;
}

double MonteCarlo::RandNumber()
{
    return dist(gen);
}

void MonteCarlo::SetBox( const py::array_t<float>& box )
{
    auto r = box.unchecked<1>();
    for ( size_t i = 0; i < dim; i++ )
        this->box[i] = r(i);

    if ( isNeighbourList )
        neighbourList.initList(numberOfParticles, rCut/this->box[0], verbose);
}

void MonteCarlo::SetBox( const std::vector<double>& box )
{
    for ( size_t i = 0; i < dim; i++ )
        this->box[i] = box[i];

    if ( isNeighbourList )
        neighbourList.initList(numberOfParticles, rCut/this->box[0], verbose);
}

void MonteCarlo::SetInitStep( const size_t n )
{
    nInit = n;
}

void MonteCarlo::InitPosition()
{
    double volume  = double( box[0] * box[1] * box[2] );
    double density = std::pow( double( numberOfParticles / volume), 1.0/3.0 );
    std::vector<int> cells(dim, 0);

    // Define a small constant epsilon to avoid floating-point precision issues
    const double epsilon = 1e-10;
    for ( size_t i = 0; i < dim; i++ )
        cells[i] = static_cast<int>( std::ceil(density*box[i] - epsilon) );
        // Calculate the number of grid cells in each dimension:
        // 1. density * box[i]: Compute the theoretical number of grid cells based on density and box size.
        // 2. Subtract epsilon: Avoid errors caused by floating-point precision when using std::ceil.
        // 3. std::ceil: Round up to ensure the grid cells fully cover the box.
        // 4. static_cast<int>: Convert the result to an integer type and store it in cells[i].

    const std::array<double, 3> gap = {
        1.0 / cells[0],
        1.0 / cells[1],
        1.0 / cells[2]
    };

    // print value of cells
    if (verbose) {
        std::cout << "cells: " << cells[0] << " " << cells[1] << " " << cells[2] << std::endl;
    }

    int n = 0; // index of particles
    bool breakFlag = false;

    for (size_t i = 0; i < cells[0] && !breakFlag; ++i)
    {
        for (size_t j = 0; j < cells[1] && !breakFlag; ++j)
        {
            for (size_t k = 0; k < cells[2] && !breakFlag; ++k)
            {
                if (n >= numberOfParticles)
                {
                    breakFlag = true;
                    break;
                }

                // calculate the position of the particle ([-0.5, 0.5])
                position[n] = {
                    (i + 0.5) * gap[0] - 0.5,
                    (j + 0.5) * gap[1] - 0.5,
                    (k + 0.5) * gap[2] - 0.5
                };

                ++n; // move to the next particle
            }
        }
    }

    if ( isNeighbourList )
    {
        neighbourList.makeList(this->position);
    }

    // std::cout << "rCut/box = " << rCut/this->box[0] << std::endl;

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

}

void MonteCarlo::SetPosition(const py::array_t<float>& position)
{
    auto r = position.unchecked<2>();
    for ( py::size_t i = 0; i < r.shape(0); i++)
        for ( py::size_t j = 0; j < r.shape(1); j++)
            this->position[i][j] = r(i, j);

    if ( isNeighbourList )
        neighbourList.makeList(this->position);

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

    for (size_t i = 0; i < numberOfParticles; i++)
        for ( size_t j = 0; j < dim; j++ )
            view(i, j) = this->position[i][j];

    return r;
}

PotentialType MonteCarlo::calculateInteraction(const std::vector<double>& p1, const std::vector<double>& p2)
{

    double r     = 0.0;
    double dr    = 0.0;
    double rCut2 = rCut * rCut;

    double sr2, sr6, sr12;

    PotentialType partial; // Partial potential and virial.
    for ( size_t i = 0; i < dim; i++ )
    {
        dr = p1[i] - p2[i];
        dr = dr - round( dr ); // Periodic boundary condition.
        dr = dr * box[i];      // Set the dr as real distance.
        r  = r + dr * dr;
    }

    sr2 = 1. / r;
    if ( sr2 > sr2Over )
    {
        partial.overlap = true;
        return partial;
    }

    if ( r < rCut2 )
    {
        sr6  = pow( sr2, 3 );
        sr12 = sr6 * sr6;

        partial.pot = sr12 - sr6;
        partial.vir = partial.pot + sr12;

        partial.pot = partial.pot * 4.0;
        partial.vir = partial.vir * 24.0 / 3.0;
    }

    return partial;
    // The unit of partial is kb * T
}

PotentialType MonteCarlo::calculateTotalPotential()
{
    PotentialType partial;

    if (isNeighbourList)
    {
        for (size_t i = 0; i < numberOfParticles; i++)
        {
            std::vector<int> ci = neighbourList.c_index(position[i]);
            std::vector<int> neighbor = neighbourList.getNeighbor(i, ci, true);
            for (auto j : neighbor)
            {
                if ( i == j ) continue; // Skip self.
                partial += calculateInteraction( position[i], position[j] );
                if ( partial.overlap )
                    return partial;
            }
        }
    }
    else
    {
        for ( size_t i = 0; i < numberOfParticles; i++ )
        {
            for ( size_t j = i + 1; j < numberOfParticles; j++ )
            {
                partial += calculateInteraction( position[i], position[j] );
                if ( partial.overlap )
                    return partial;
            }
        }
    }
    return partial;
}

py::object MonteCarlo::GetPotential()
{
    PotentialType partial;
    auto potential = py::dict();

    partial = calculateTotalPotential();

    potential["pot"] = partial.pot;
    potential["vir"] = partial.vir;
    potential["overlap"] = partial.overlap;

    return potential;
}

py::object MonteCarlo::MoveParticles( const double drMax )
{
    auto potential = py::dict();
    auto pot = displacementParticles( drMax );

    potential["pot"] = pot["pot"];
    potential["vir"] = pot["vir"];
    potential["moves"] = pot["moves"];

    return potential;
}

std::map<std::string, double> MonteCarlo::displacementParticles( double drMax )
{
    std::map<std::string, double> result;
    int move = 0;

    std::vector<double> pos(3, 0);
    std::vector<double> poso(3, 0);

    PotentialType total; // total potential
    double delta;

    int moves = 0;
    for ( size_t i = 0; i < numberOfParticles; i++ )
    {
        for ( int j = 0; j < dim; j++ )
        {
            pos[j]  = position[i][j] + ( 2*RandNumber() - 1 ) * drMax;
            poso[j] = position[i][j];

            pos[j] = pos[j] - round( pos[j] ); // pbc
        }

        PotentialType partial0, partial1;

        if (isNeighbourList)
        {
            std::vector<int> ci = neighbourList.c_index(position[i]);
            std::vector<int> neighbor = neighbourList.getNeighbor(i, ci, false);

            // calculate potential and virial for old position
            for ( auto k : neighbor )
            {
                if ( i == k ) continue;
                partial0 += calculateInteraction( poso,  position[k] );
                if ( partial0.overlap )
                {
                    std::cout << "Error: overlap, displacement particles." << std::endl;
                    exit(-1);
                }
            }

            // calculate potential and virial for new position
            ci = neighbourList.c_index(pos);
            neighbourList.moveInList( i, ci );
            neighbor = neighbourList.getNeighbor(i, ci, false);

            for ( auto k : neighbor )
            {
                if ( i == k ) continue;
                partial1 += calculateInteraction( pos,  position[k] );
                if ( partial1.overlap )
                    break;
            }
        }
        else
        {
            for ( size_t k = 0; k < numberOfParticles; k++ )
            {
                if ( i == k ) continue;
                partial1 += calculateInteraction( pos,  position[k] );
                partial0 += calculateInteraction( poso, position[k] );

                if ( partial0.overlap )
                {
                    std::cout << "Error: overlap, displacement particles." << std::endl;
                    exit(-1);
                }

                if ( partial1.overlap )
                    break;
            }
        }

        if ( !(partial1.overlap) )
        {
            delta = partial1.pot - partial0.pot;
            delta = delta / temperature;

            if ( metropolis( delta ) )
            {
                for ( size_t j = 0; j < dim; j++ )
                    position[i][j] = pos[j];
                total += ( partial1- partial0 );
                moves += 1;
            }
            else if(isNeighbourList)
            {
                // move the ith particle back to its original position
                auto ci = neighbourList.c_index(position[i]);
                neighbourList.moveInList(i, ci);
            }
        }
        else
        {
            if ( isNeighbourList ) 
            {
                // move the ith particle back to its original position
                auto ci = neighbourList.c_index(position[i]);
                neighbourList.moveInList(i, ci);
            }

        }
    }

    result["pot"]   = total.pot;
    result["vir"]   = total.vir;
    result["moves"] = moves;

    return result;
}

bool MonteCarlo::metropolis( double delta )
{
    double trial = 0.0;
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

// ========== GCA methods ========== //

void MonteCarlo::GetLJDiff(const std::vector<double>& r_old, const std::vector<double>& r_new, const std::vector<double>& r_j, double& d_pot, double& d_vir)
{
    double rCutSq = rCut * rCut;

    // lambda: Calculate the single point energy
    auto calc = [&](const std::vector<double>& pos_i, const std::vector<double>& pos_j) -> std::pair<double, double>
    {
        double r2 = 0.0;
        for ( size_t k = 0; k < dim; k++)
        {
            double diff = pos_i[k] - pos_j[k];
            diff -= round(diff); // PBC [-0.5, 0.5]
            diff *= box[k];
            r2 += diff * diff;
        }

        if (r2 >= rCutSq) return {0.0, 0.0};
        if (r2 < 1e-10) r2 = 1e-10;

        double inv2 = 1.0 / r2;
        double inv6 = inv2 * inv2 * inv2;
        double inv12 = inv6 * inv6;

        // return the potential and Virial
        return {4.0 * (inv12 - inv6), 8.0 * (2.0 * inv12 - inv6)};
    };

    auto [p1, v1] = calc(r_old, r_j);
    auto [p2, v2] = calc(r_new, r_j);
    d_pot = p2 - p1;
    d_vir = v2 - v1;
}

MonteCarlo::GCAResult MonteCarlo::StepGCA()
{
    // Select Pivot and Seed
    std::vector<double> pivot(dim);
    for (int k = 0; k < dim; k++) pivot[k] = RandNumber() - 0.5;

    int seed = (int)(RandNumber() * numberOfParticles);
    if (seed >= numberOfParticles)
        seed = numberOfParticles - 1;

    // Reset the state of the system
    std::fill(m_InCluster.begin(), m_InCluster.end(), false);
    std::fill(m_BoundaryPotAcc.begin(), m_BoundaryPotAcc.end(), 0.0);
    std::fill(m_BoundaryVirAcc.begin(), m_BoundaryVirAcc.end(), 0.0);
    m_Stack.clear();

    m_Stack.push_back(seed);
    m_InCluster[seed] = true;

    double totalDPot = 0.0;
    double totalDVir = 0.0;
    int cSize = 0;

    // Cluster growth
    while (!m_Stack.empty())
    {
        int i = m_Stack.back();
        m_Stack.pop_back();
        cSize++;

        std::vector<double> r_old = position[i];
        std::vector<double> r_new(dim);

        // reverse cordinate
        for (int k = 0; k < dim; k++)
        {
            double val = 2.0 * pivot[k] - r_old[k];
            val -= round(val); // PBC
            r_new[k] = val;
            position[i][k] = val; // Renewal of position
        }

        // Renew the position, by using linkList (O(1))
        neighbourList.update(i, r_new);

        // Get the neighbors (from old and new position)
        m_CandidateList.clear();
        neighbourList.getCandidatesFromPos(r_old, m_CandidateList);
        neighbourList.getCandidatesFromPos(r_new, m_CandidateList);

        for(int j : m_CandidateList)
        {
            if (i == j) continue;
            if (m_InCluster[j]) continue;
            if (m_IsCandidate[j]) continue;

            m_IsCandidate[j] = true;

            double dp, dv;
            GetLJDiff(r_old, r_new, position[j], dp, dv);

            bool join = false;
            if (dp > 0)
            {
                double prob = 1.0 - std::exp(-dp / temperature);
                if (prob > RandNumber()) join = true;
            }

            if (join)
            {
                totalDPot -= m_BoundaryPotAcc[j];
                totalDVir -= m_BoundaryVirAcc[j];
                m_BoundaryPotAcc[j] = 0.0;
                m_BoundaryVirAcc[j] = 0.0;

                m_InCluster[j] = true;
                m_Stack.push_back(j);
            }
            else
            {
                totalDPot += dp;
                totalDVir += dv;
                m_BoundaryPotAcc[j] += dp;
                m_BoundaryVirAcc[j] += dv;
            }
        }

        // Reset the state of m_IsCandidate: set all positions to false.
        for (int j : m_CandidateList) m_IsCandidate[j] = false;
    }

    return {totalDPot, totalDVir, cSize};
}

// =============================================== //

double MonteCarlo::testParticles()
{
    PotentialType partial;

    std::vector<double> pos(3,0);
    for ( size_t j = 0; j < dim; j++ )
        pos[j] = RandNumber() - 0.5;

    for ( size_t i = 0; i < numberOfParticles; i++)
    {
        partial += calculateInteraction( pos, position[i] );
        if ( partial.overlap )
            return INFINT;
    }

    return partial.pot;
}

//py::array_t<float> MonteCarlo::NVTrun( int nStep, double drMax )
std::map<std::string, std::vector<double>> MonteCarlo::NVTrun( int nStep, double drMax, int interval )
{

    std::map<std::string, std::vector<double>> pot;
    pot.insert(std::pair<std::string, std::vector<double>>("potential",  std::vector<double>(nStep, 0)));
    pot.insert(std::pair<std::string, std::vector<double>>("vir",        std::vector<double>(nStep, 0)));
    pot.insert(std::pair<std::string, std::vector<double>>("moveRatios", std::vector<double>(nStep, 0)));
    pot.insert(std::pair<std::string, std::vector<double>>("chemicalP",  std::vector<double>(nStep, 0)));

    if (verbose) {
        std::cout << "Initial Start" << std::endl;
    }
    for (size_t i = 0; i < nInit; i++)
    {
        if (m_Method == SimulationMethod::METROPOLIS)
        {
            auto results = displacementParticles(drMax);

            double moveRatio = results["moves"] / numberOfParticles;
            if ( moveRatio > 0.55 )
                drMax *= 1.05;
            else if (moveRatio < 0.45)
                    drMax *= 0.95;
        }
        else if (m_Method == SimulationMethod::GCA)
        {
            GCAResult res = StepGCA();
        }
        else
        {
            std::cerr << "Error: Simulation method not found or not implemented." << std::endl;
            exit(-1);
        }
    }
    if (verbose) {
        std::cout << "Initial Done." << std::endl;
    }


    PotentialType partial = calculateTotalPotential();
    for (size_t i = 0; i < nStep; i++)
    {

        std::map<std::string, double> results;

        if (m_Method == SimulationMethod::METROPOLIS)
        {
            results = displacementParticles(drMax);
            /*
            if (i%1000==0)
                std::cout << i << std::endl;
            */

            double moveRatio = results["moves"] / numberOfParticles;
            if (moveRatio > 0.55)
                drMax *= 1.05;
            else if (moveRatio < 0.45)
                drMax *= 0.95;

            pot["moveRatios"][i] = moveRatio;

        }
        else if (m_Method == SimulationMethod::GCA)
        {
            GCAResult res = StepGCA();
            results["pot"] = res.d_pot;
            results["vir"] = res.d_vir;
            results["moves"] = res.clusterSize;

            pot["moveRatios"][i] = res.clusterSize;
        }
        else
        {
                std::cerr << "Error: Simulation method not found or not implemented." << std::endl;
                exit(-1);
        }

        if ( i == 0)
        {
            pot["vir"][i]       = partial.vir + results["vir"];
            pot["potential"][i] = partial.pot + results["pot"];
        }
        else
        {
            pot["vir"][i]       = pot["vir"][i-1]       + results["vir"];
            pot["potential"][i] = pot["potential"][i-1] + results["pot"];
        }

        double pot1 = testParticles();
        pot1 = exp( - pot1 / temperature );
        pot["chemicalP"][i] = pot1;

        // write trajectory to XYZ file
        if (i%interval == 0)
        {
            if ( i == 0)
                writeXYZTrajectory(filename, i, false);
            else
                writeXYZTrajectory(filename, i, true);
        }

    }

    return pot;

    /*
    int shape[2]{nStep, 3};
    auto r    = py::array_t<float>(shape);
    auto view = r.mutable_unchecked<2>();

    int n = 0;
    for ( auto x : pot )
    {
        int m = 0;
        for ( auto y : x.second )
            view(n, m++) = y;

        n++;
    }

    return r;
    */
}

void MonteCarlo::writeXYZTrajectory( const std::string& filename, int timestep,bool append )
{
    std::ofstream file;
    if (append)
        file.open(filename, std::ios_base::app);
    else
        file.open(filename);

    if ( !file.is_open() )
        throw std::runtime_error("Could not open file " + filename);

    file << numberOfParticles << "\n";
    file << "Step " << timestep << "\n";

    for ( auto& p : position )
    {
        file << "Ar " << " ";
        for ( auto& x : p )
            file << x << " ";
        file << "\n";
    }
    file.close();
}