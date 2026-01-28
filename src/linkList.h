#ifndef LINKLIST_H
#define LINKLIST_H
#include <vector>
#include <iostream>
#include <cmath>

class linkList
{
    public:
    linkList() = default;
    ~linkList();
    void initList( const int numberOfParticle, const double rCutBox, bool verbose = false );
    void makeList( const std::vector<std::vector<double>>& r );
    std::vector<int> c_index( const std::vector<double>& r );
    void creatInList( const int, const std::vector<int>& );
    void destroyInList( const int, const std::vector<int>& );
    void moveInList( const int, const std::vector<int>& );
    std::vector<int> getNeighbor( const int, const std::vector<int>&, bool );

    // ========== For GCA ========== //

    // Update the particle position
    void update(int p_idx, const std::vector<double>& r_new);

    // Get the number of particles; the input is a position, not a particle index
    void getCandidatesFromPos(const std::vector<double>& r, std::vector<int>& results);

    // Set verbose mode
    void setVerbose(bool v) { verbose = v; }

    // ============================= //

    private:

    int numberOfParticles;
    // assume is a cubic box, and this virable is equal to (Cut of length) / (Box length)
    double rCutBox;
    // number of cells in each dimension
    int sc;

    const int dim = 3;

    // Verbose flag for output control
    bool verbose = false;

    std::vector<std::vector<int>> c;
    // size is (numberOfParticles, dim), to store neighbour cell index of each particles
    std::vector<std::vector<std::vector<int>>> head;
    // size is (dim, dim, dim), to store the head of each cell
    std::vector<int> list;
    // size is (numberOfParticles), to store the neighbour cell list. -1 reperesent the end of the list

    const std::vector<std::vector<int>> neighborBox = {\
        {-1, -1, -1}, { 0, -1, -1}, { 1, -1, -1},\
        {-1,  0, -1}, { 0,  0, -1}, { 1,  0, -1},\
        {-1,  1, -1}, { 0,  1, -1}, { 1,  1, -1},\
        {-1, -1,  0}, { 0, -1,  0}, { 1, -1,  0},\
        {-1,  0,  0}, { 0,  0,  0}, { 1,  0,  0},\
        {-1,  1,  0}, { 0,  1,  0}, { 1,  1,  0},\
        {-1, -1,  1}, { 0, -1,  1}, { 1, -1,  1},\
        {-1,  0,  1}, { 0,  0,  1}, { 1,  0,  1},\
        {-1,  1,  1}, { 0,  1,  1}, { 1,  1,  1}\
    };

};

#endif // LINKLIST_H