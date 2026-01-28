"""
Test radial distribution function (gr) calculation.

This test verifies the calculation of radial distribution function
using Metropolis algorithm and compares GCA with Metropolis.
Reference: MC_NVT.ipynb for plotting style.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))

import numpy as np
import matplotlib.pyplot as plt

try:
    import MonteCarlo as MonteCarlo_py
except ImportError:
    print("Python MonteCarlo module not available, skipping test")
    sys.exit(0)

try:
    import build.MonteCarlo as MonteCarlo_cpp
except ImportError:
    print("C++ MonteCarlo module not available, skipping test")
    sys.exit(0)


def test_radial_distribution_metropolis():
    """Test radial distribution function using Metropolis algorithm.
    
    Reference: MC_NVT.ipynb for plotting style.
    """
    print("\n=== Testing Metropolis Algorithm (Density 0.8442, Temperature 1.5043) ===")
    
    # Load reference data from gr.dat (same as MC_NVT.ipynb)
    try:
        g_data = np.loadtxt("../data/gr.dat", skiprows=1)
        print(f"Loaded reference data from gr.dat: {len(g_data)} points")
    except Exception as e:
        print(f"Warning: Could not load gr.dat: {e}")
        g_data = None
    
    # Configuration for known data comparison
    density = 0.8442
    temperature = 1.5043
    # Use cells initialization since 512 = 8×8×8
    cells = np.array([8, 8, 8])
    num = np.prod(cells)  # 512 particles
    rCut = 2.5
    
    system = {
        "cells": cells,
        "density": density,
        "numberOfParticles": num,
        "temperature": temperature,
        "rCut": rCut,
        "drMax": 0.25,
        "Version": "CPP",
        "isNeighborList": True,
        "initStep": 1000,
        "method": "metropolis"
    }
    system["size"] = system["cells"] / ( system["density"] ) ** (1./3.)
    
    nBlock = 20
    nStep = 1000
    
    # Run simulation
    mc = MonteCarlo_py.MonteCarlo(system)
    pressures, moveRatios, totalEnergy, cp, poss = mc.MC_NVT(nBlock, nStep)
    _, _, _, _, _ = mc.MC_NVT_CPP(nBlock, nStep)
    
    # Calculate radial distribution function from positions
    rMid, g = MonteCarlo_py.MonteCarlo.rdf( poss[:], system["size"][0], 512, dr=0.02 )

    # Calculate radial distribution function from positions, CPP version
    positions = MonteCarlo_py.MonteCarlo.readXYZTrajectory("trajectory.xyz")
    rMid1, g1 = MonteCarlo_py.MonteCarlo.rdf( positions[:], system["size"][0], 512, dr=0.02 )
    
    # Plot results (same style as MC_NVT.ipynb)
    print("\n=== Plotting Results ===")
    plt.figure(figsize=(8, 6))
    
    # Plot MC simulation results
    plt.plot(rMid, g, color="b", label="MC Metropolis")
    plt.plot(rMid1, g1, color="r", label="MC CPP")
    
    # Plot reference data if available
    if g_data is not None:
        plt.scatter(g_data[:,0], g_data[:,1], marker="o",
                   edgecolors="r", color='none', label="Reference (Frenkel)")
    
    plt.xlabel(r"$r/\sigma$")
    plt.ylabel("g(r)")
    plt.title(f"Radial Distribution Function (Density={density}, T={temperature})")
    
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Save plot to file
    plot_filename = "radial_distribution_metropolis.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")
    
    # Show plot (if running interactively)
    plt.show()


def test_radial_distribution_gca_vs_metropolis():
    """Compare radial distribution function between GCA and Metropolis.
    
    Reference: MC_NVT.ipynb for plotting style.
    """
    print("\n=== Comparing GCA vs Metropolis (Density 0.32, Temperature 1.18) ===")
    
    # Configuration for GCA vs Metropolis comparison
    density = 0.32
    temperature = 1.18
    # Use cells initialization since 512 = 8×8×8
    cells = np.array([8, 8, 8])
    num = np.prod(cells)  # 512 particles
    rCut = 2.5
    
    nBlock = 20
    nStep = 1000
    
    # Metropolis
    print("\n--- Metropolis ---")
    system_metropolis = {
        "cells": cells,
        "density": density,
        "numberOfParticles": num,
        "temperature": temperature,
        "rCut": rCut,
        "drMax": 0.25,
        "Version": "CPP",
        "isNeighborList": True,
        "initStep": 1000,
        "method": "metropolis"
    }
    system_metropolis["size"] = system_metropolis["cells"] / ( system_metropolis["density"] ) ** (1./3.)

    mc_metropolis = MonteCarlo_py.MonteCarlo(system_metropolis)
    pressures_met, moveRatios_met, totalEnergy_met, cp_met, poss_met = mc_metropolis.MC_NVT_CPP(nBlock, nStep)
    
    # Calculate radial distribution function from positions
    positions = MonteCarlo_py.MonteCarlo.readXYZTrajectory("trajectory.xyz")
    rMid, g = MonteCarlo_py.MonteCarlo.rdf( positions[:], system_metropolis["size"][0], 512, dr=0.02 )

    # Delete the xyz file
    os.remove("trajectory.xyz")

    
    # GCA
    print("\n--- GCA ---")
    system_gca = {
        "cells": cells,
        "density": density,
        "numberOfParticles": num,
        "temperature": temperature,
        "rCut": rCut,
        "drMax": 0.25,
        "Version": "CPP",
        "isNeighborList": True,
        "initStep": 2000,
        "method": "gca"
    }
    system_gca["size"] = system_gca["cells"] / ( system_gca["density"] ) ** (1./3.)
    
    mc_gca = MonteCarlo_py.MonteCarlo(system_gca)
    pressures_gca, moveRatios_gca, totalEnergy_gca, cp_gca, poss_gca = mc_gca.MC_NVT_CPP(nBlock, nStep)
    
    # Calculate radial distribution function from positions
    positions1 = MonteCarlo_py.MonteCarlo.readXYZTrajectory("trajectory.xyz")
    rMid1, g1 = MonteCarlo_py.MonteCarlo.rdf( positions1[:], system_gca["size"][0], 512, dr=0.02 )

    # Delete the xyz file
    os.remove("trajectory.xyz")
    
    # Plot results (same style as MC_NVT.ipynb)
    print("\n=== Plotting Results ===")
    plt.figure(figsize=(8, 6))
    
    # Plot both algorithms
    plt.plot(rMid, g, color="b", label="Metropolis")
    plt.plot(rMid1, g1, color="r", label="GCA")
    
    plt.xlabel(r"$r/\sigma$")
    plt.ylabel("g(r)")
    plt.title(f"Radial Distribution Function Comparison (Density={density}, T={temperature})")
    
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Save plot to file
    plot_filename = "radial_distribution_gca_vs_metropolis.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")
    
    # Show plot (if running interactively)
    plt.show()


if __name__ == "__main__":
    test_radial_distribution_metropolis()
    test_radial_distribution_gca_vs_metropolis()
