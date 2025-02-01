#include "linkList.h"

void linkList::initList( const int numberOfParticles, const double rCutBox )
{
    this->numberOfParticles = numberOfParticles;
    this->rCutBox           = rCutBox;

    this->sc = std::floor( 1.0 / rCutBox );
    std::cout << "sc = " << sc << std::endl;

    // create the neighbour cell list, to store the cell index of each particle
    c.resize(numberOfParticles);
    for ( auto& it : c )
        it.resize( dim, -1 );

    // create the link list
    list.resize( numberOfParticles, -1 );

    // create the head list
    head.resize(sc);
    for ( auto& it : head )
    {
        it.resize(sc);
        for ( auto& it2 : it )
        {
            it2.resize(sc, -1);
        }
    }
}

linkList::~linkList()
{
    c.clear();
    list.clear();
    head.clear();
}

void linkList::makeList( const std::vector<std::vector<double>>& r )
{
    std::vector<int> ci(dim, 0.0);
    // create the link list
    for ( size_t i = 0; i < numberOfParticles; i++ )
    {
        // find the neighbour cell index of each particle
        ci = c_index( r[i] );
        // adding the ith particle to the link list
        creatInList( i, ci );
    }

    /* debug
    for (size_t i = 0; i < numberOfParticles; i++)
    {
        std::cout << "r[i] = ";
        for ( auto x : r[i] )
            std::cout << x << '\t';
        // find the neighbour cell index of each particle
        ci = c_index( r[i] );
        std::cout << "c_index = ";
        for ( auto x : ci )
            std::cout << x << '\t';
        std::cout << "c[i] = ";
        for ( auto x : c[i] )
            std::cout << x << '\t';
        std::cout << std::endl;
    }
    */

    // int j = 9;
    // std::vector<int> cj = {0,0,0};
    // moveInList( j, cj );
    // for ( auto x : list )
    //     std::cout << x << std::endl;

    // std::cout << "head = " << std::endl;
    // for ( auto& x : head )
    // {
    //     for ( auto& y : x )
    //     {
    //         for ( auto& z : y )
    //         {
    //             std::cout << z << std::endl;
    //         }
    //     }
    // }

    // auto neighbor = getNeighbor( j, cj, false);
    // std::cout << "neighbor = " << std::endl;
    // for ( auto x : neighbor )
    //     std::cout << x << std::endl;
}

std::vector<int> linkList::c_index( const std::vector<double>& r )
{
    std::vector<int> ci(dim, 0);
    for ( size_t i = 0; i < dim; i++ )
    {
        ci[i] = std::floor( (r[i] + 0.5) * double(sc) );
        // std::cout << ci[i] << std::endl;

        // using periodic boundary conditions
        if (ci[i] >= sc) ci[i] = sc - 1;
        if (ci[i] < 0) ci[i] = 0;
    }

    return ci;
}

void linkList::creatInList( const int i, const std::vector<int>& ci )
{
    // let the ith particle point to the current head of this neighbour cell
    list[i] = head[ci[0]][ci[1]][ci[2]];
    // let the ith particle be the new head of this neighbour cell
    head[ci[0]][ci[1]][ci[2]] = i;
    for ( size_t j = 0; j < dim; j++)
        c[i][j] = ci[j]; // store the neighbour cell index of the ith particle
}

void linkList::destroyInList(const int i, const std::vector<int>& ci)
{
    if ( head[ci[0]][ci[1]][ci[2]] == i )
    {
        // if the ith particle is the head of the list
        head[ci[0]][ci[1]][ci[2]] = list[i];
    }
    else
    {
        // if the ith particle is not the head of the list
        int j = head[ci[0]][ci[1]][ci[2]];
        while ( list[j] != i )
            j = list[j];
        list[j] = list[i];
    }
}

void linkList::moveInList(const int i, const std::vector<int>& ci)
{
    if ( c[i][0] == ci[0] && c[i][1] == ci[1] && c[i][2] == ci[2] )
        return; // if the ith particle is already in the same cell, do nothing

    destroyInList(i, c[i]); // destroy the ith particle in the list
    creatInList(i, ci);     // create the ith particle in the list
}

std::vector<int> linkList::getNeighbor(const int i, const std::vector<int>& ci, bool half)
{
    std::vector<int> neighbor;
    std::vector<int> cj(dim, -1);
    int k1 = 0, k2;
    int j;

    if (half)
        k2 = 14; // if half is true, only consider the 13 neighbor cells and only the down-list particles in the same cell
    else
        k2 = 27; // if half is false, consider the 26 neighbor cells and all the particles in the same cell

    for ( int k = k1; k < k2; k++)
    {
        for ( int m = 0; m < dim; m++ )
        {
            cj[m] = ci[m] + neighborBox[k][m];
            if ( cj[m] < 0   ) cj[m] += sc;
            if ( cj[m] >= sc ) cj[m] -= sc;
        }

        if ( cj[0] == ci[0] && cj[1] == ci[1] && cj[2] == ci[2] && half )
            j = list[i]; // only consider the down-list particles in the same cell
        else
            j = head[cj[0]][cj[1]][cj[2]];

        while ( j != -1 )
        {
            if ( j != i )
                neighbor.push_back(j);
            j = list[j];
        }
    }

    return neighbor;
}
