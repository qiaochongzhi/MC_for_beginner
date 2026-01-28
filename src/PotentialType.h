#ifndef POTENTIALTYPE_H
#define POTENTIALTYPE_H
#pragma once

class PotentialType
{
public:
    double pot  = 0;
    double vir  = 0;
    bool overlap = false;

    PotentialType(double pot = 0., double vir = 0., bool overlap = 0.) : pot(pot), vir(vir), overlap(overlap) {}
    ~PotentialType() {}

    void reset() { pot = 0; vir = 0; overlap = false; }
    PotentialType operator+(const PotentialType& other) const
    {
        return PotentialType(pot + other.pot, vir + other.vir, overlap || other.overlap);
    }
    PotentialType& operator+=(const PotentialType& other)
    {
        pot += other.pot;
        vir += other.vir;
        overlap = overlap || other.overlap;
        return *this;
    }
    PotentialType operator-(const PotentialType& other) const
    {
        return PotentialType(pot - other.pot, vir - other.vir, overlap || other.overlap); // overlap is not subtracted
    }

    PotentialType& operator=(const PotentialType& other)
    {
        pot = other.pot;
        vir = other.vir;
        overlap = other.overlap;
        return *this;
    }
};

#endif // POTENTIALTYPE_H