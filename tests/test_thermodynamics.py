"""
Test thermodynamic properties: pressure, internal energy, and chemical potential.

This test verifies the calculation of pressure, internal energy, and chemical potential
for different densities using both Metropolis and GCA algorithms.
"""

import sys
import os
import multiprocessing as mp
from functools import partial
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))

import numpy as np
import matplotlib.pyplot as plt

try:
    import MonteCarlo as MonteCarlo_py
    from MBWR_EOS import MBWR_EOS
except ImportError:
    print("Python MonteCarlo module not available, skipping test")
    sys.exit(0)

try:
    import build.MonteCarlo as MonteCarlo_cpp
except ImportError:
    print("C++ MonteCarlo module not available, skipping test")
    sys.exit(0)

# Load chemPex.dat data
testChemPex = np.loadtxt(os.path.join(os.path.dirname(__file__), '../data/chemPex.dat'), skiprows=1)


def uTail(rho, rCut):
    """Calculate tail correction for energy."""
    tail = (8./3.) * np.pi * rho * ( (1./3.) * (1./rCut)**9 - (1./rCut)**3 )
    return tail


def run_simulation_for_density(system, num_particles, nBlock=20, nStep=1000):
    """Run simulation for a specific density."""
    system["numberOfParticles"] = num_particles
    MC = MonteCarlo_py.MonteCarlo(system)
    p,  m,  t,  pot,  _ = MC.MC_NVT(nBlock=nBlock, nStep=nStep)
    p1, m1, t1, pot1, mu2 = MC.MC_NVT_CPP(nBlock=nBlock, nStep=nStep)
    
    return {
        "density": num_particles / np.prod(system["size"]),
        "num_particles": num_particles,
        "pressures_py": p,
        "moveRatios_py": m,
        "totalEnergy_py": t,
        "cp_py": pot,
        "pressures_cpp": p1,
        "moveRatios_cpp": m1,
        "totalEnergy_cpp": t1,
        "cp_cpp": pot1,
        "chemP_cpp": mu2
    }


def test_thermodynamics_metropolis():
    """Test thermodynamic properties using Metropolis algorithm."""
    print("\n=== Testing Metropolis Algorithm ===")
    
    system = {
        "size": np.array([10, 10, 10]),
        "temperature": 2.0,
        "rCut": 2.5,
        "drMax": 0.25,
        "Version": "CPP",
        "isNeighborList": True,
        "initStep": 1000,
        "method": "metropolis"
    }
    
    nBlock = 20
    nStep = 1000
    
    # Density sweep: 100-900 particles (0.1-0.9 density)
    testDensity = np.arange(100, 1000, 100)
    
    # Use multiprocessing to run simulations in parallel
    print("Running simulations in parallel...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(partial(run_simulation_for_density, system, nBlock=nBlock, nStep=nStep), testDensity)
    
    # Organize results
    pressures_py, moveRatios_py, totalEnergy_py, cp_py = [], [], [], []
    pressures_cpp, moveRatios_cpp, totalEnergy_cpp, cp_cpp, chemP_cpp = [], [], [], [], []
    
    for result in results:
        print(f"\n--- Density {result['density']:.3f} ({result['num_particles']} particles) ---")
        print(f"Pressure (Python): {np.mean(np.array(result['pressures_py'][2000:])):.4f}")
        print(f"Pressure (C++): {np.mean(np.array(result['pressures_cpp'])):.4f}")
        
        pressures_py.append(result['pressures_py'])
        moveRatios_py.append(result['moveRatios_py'])
        totalEnergy_py.append(result['totalEnergy_py'])
        cp_py.append(result['cp_py'])
        
        pressures_cpp.append(result['pressures_cpp'])
        moveRatios_cpp.append(result['moveRatios_cpp'])
        totalEnergy_cpp.append(result['totalEnergy_cpp'])
        cp_cpp.append(result['cp_cpp'])
        chemP_cpp.append(result['chemP_cpp'])
    
    # Calculate and test the system pressure
    print("\n=== Pressure Calculation ===")
    presCPP = [np.mean(np.array(i)) for i in pressures_cpp]
    presPY = [np.mean(np.array(i[2000:])) for i in pressures_py]
    
    xx2 = np.arange(0.1, 1.0, 0.1)
    
    # Data from MBWR paper 89 
    xx1 = np.array([0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    yy1 = np.array([0.1777, 0.3290, 0.705, 1.069, 1.756, 3.024, 5.28, 9.09])
    
    plt.figure(figsize=(8, 6))
    plt.plot(xx1, yy1, color="b", linestyle=":", marker="", label="MBWR")
    plt.plot(xx2, presPY, color="g", linestyle="", marker="1", markersize=12, label="Python")
    plt.plot(xx2, presCPP, color="r", linestyle="", marker="2", markersize=12, label="C++")
    
    plt.xlabel("Density, " + r"$\rho \sigma^3$")
    plt.ylabel("Pressure, " + r"$P$, unit in $k_B T$")
    
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("pressure_metropolis.png")
    #plt.show()
    
    # Calculate and test the total energy
    print("\n=== Total Energy Calculation ===")
    totU = [
        [0.1, -0.669], [0.2, -1.308], [0.3, -1.922], [0.4, -2.539], [0.5, -3.149],
        [0.6, -3.747], [0.7, -4.3], [0.8, -4.752], [0.9, -5.025]
    ]
    totU = np.array(totU)
    
    num = np.arange(100, 1000, 100)
    up, uc = [], []
    
    for i, t in enumerate(totalEnergy_cpp):
        u1 = np.mean(np.array(t)) / num[i]
        u2 = np.mean(np.array(totalEnergy_py[i][1000:])) / num[i]
        uc.append(u1)
        up.append(u2)
    
    plt.figure(figsize=(8, 6))
    plt.plot(xx2, np.array(uc), color="r", linestyle="", marker="1", markersize=12, label="C++")
    plt.plot(xx2, np.array(up), color="g", linestyle="", marker="2", markersize=12, label="Python")
    plt.plot(totU[:,0], totU[:,1], color="b", linestyle="--", marker="", label="Reference")
    
    plt.xlabel("Density, " + r"$\rho \sigma^3$")
    plt.ylabel("Energy per particle, " + r"$U/N$")
    
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("energy_metropolis.png")
    #plt.show()
    
    # Calculate and test chemical potential
    print("\n=== Chemical Potential Calculation ===")
    import MBWR_EOS
    
    test = {"temperature": 2.0, "sigma": 1., "component": 1, "epsilon": 1., "wave": 1.0}
    CP = []
    
    xx2 = np.arange(0.1, 1.0, 0.1)
    mbwr = MBWR_EOS.MBWR_EOS(test)
    for i in xx2:
        mbwr.rho = i
        chemP = mbwr.exChemicalPotential()
        CP.append(chemP[0][0])
    
    CP = np.array(CP)
    
    # data from CMC result of other program
    yy = np.array([-0.124232, -0.186052, -0.172991, -0.0584114, 0.244226, 0.771789, 1.7077])
    xx = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    
    tCPex_PY = [np.mean(np.array(i[2000:])) for i in cp_py]
    tCPex_PY = -np.log(tCPex_PY)
    
    tCPex_CPP = [np.mean(np.array(i)) for i in cp_cpp]
    tCPex_CPP = -np.log(tCPex_CPP)
    
    xx2 = np.arange(0.1, 1.0, 0.1)
    uts = []
    for i in xx2:
        ut = uTail(i, 2.5)
        uts.append(ut)
    
    tCPex_PY = tCPex_PY + (1./2.) * 2.0 * np.array(uts)  # unit: 1
    tCPex_CPP = tCPex_CPP + (1./2.) * 2.0 * np.array(uts)  # unit: 1
    
    plt.figure(figsize=(8, 6))
    plt.plot(testChemPex[:,0], testChemPex[:,1]/2., color="m", linestyle=":", marker="", label="Frenkel")
    #plt.plot(xx, yy, color="b", linestyle="", marker="s", markersize=8, label="CMC")
    plt.plot(xx2, tCPex_PY, color="g", linestyle="", marker="1", markersize=12, label="Python")
    plt.plot(xx2, tCPex_CPP, color="r", linestyle="", marker="2", markersize=12, label="C++")
    plt.plot(xx2, CP, color="b", linestyle="", marker="o", markerfacecolor='none', markersize=8, label="MBWR")
    
    plt.xlabel("Density, " + r"$\rho \sigma^3$")
    plt.ylabel("Excess Chemical Potential, " + r"$\beta \mu^{ex}$")

    plt.xlim([-0.02, 0.82])
    
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("chemical_potential_metropolis.png")
    #plt.show()


def test_thermodynamics_gca():
    """Test thermodynamic properties using GCA algorithm."""
    print("\n=== Testing GCA Algorithm ===")
    
    system = {
        "size": np.array([10, 10, 10]),
        "temperature": 2.0,
        "rCut": 2.5,
        "drMax": 0.25,
        "Version": "CPP",
        "isNeighborList": True,
        "initStep": 1000,
        "method": "gca"
    }
    
    nBlock = 20
    nStep = 1000
    
    # Density sweep: 100-600 particles (0.1-0.6 density) - limited due to efficiency
    testDensity = np.arange(100, 700, 100)
    
    # Use multiprocessing to run simulations in parallel
    print("Running simulations in parallel...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(partial(run_simulation_for_density, system, nBlock=nBlock, nStep=nStep), testDensity)
    
    # Organize results
    pressures_py, moveRatios_py, totalEnergy_py, cp_py = [], [], [], []
    pressures_cpp, moveRatios_cpp, totalEnergy_cpp, cp_cpp, chemP_cpp = [], [], [], [], []
    
    for result in results:
        print(f"\n--- Density {result['density']:.3f} ({result['num_particles']} particles) ---")
        print(f"Pressure (Python): {np.mean(np.array(result['pressures_py'][2000:])):.4f}")
        print(f"Pressure (C++): {np.mean(np.array(result['pressures_cpp'])):.4f}")
        
        pressures_py.append(result['pressures_py'])
        moveRatios_py.append(result['moveRatios_py'])
        totalEnergy_py.append(result['totalEnergy_py'])
        cp_py.append(result['cp_py'])
        
        pressures_cpp.append(result['pressures_cpp'])
        moveRatios_cpp.append(result['moveRatios_cpp'])
        totalEnergy_cpp.append(result['totalEnergy_cpp'])
        cp_cpp.append(result['cp_cpp'])
        chemP_cpp.append(result['chemP_cpp'])
    
    # Calculate and test the system pressure
    print("\n=== Pressure Calculation ===")
    presCPP = [np.mean(np.array(i)) for i in pressures_cpp]
    presPY = [np.mean(np.array(i[2000:])) for i in pressures_py]
    
    xx2 = np.arange(0.1, 0.7, 0.1)
    
    # Data from MBWR paper 89 
    xx1 = np.array([0.1, 0.2, 0.4, 0.5, 0.6])
    yy1 = np.array([0.1777, 0.3290, 0.705, 1.069, 1.756])
    
    plt.figure(figsize=(8, 6))
    plt.plot(xx1, yy1, color="b", linestyle=":", marker="", label="MBWR")
    plt.plot(xx2, presPY, color="g", linestyle="", marker="1", markersize=12, label="Python")
    plt.plot(xx2, presCPP, color="r", linestyle="", marker="2", markersize=12, label="C++")
    
    plt.xlabel("Density, " + r"$\rho \sigma^3$")
    plt.ylabel("Pressure, " + r"$P$, unit in $k_B T$")
    
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("pressure_gca.png")
    plt.show()
    
    # Calculate and test the total energy
    print("\n=== Total Energy Calculation ===")
    totU = [
        [0.1, -0.669], [0.2, -1.308], [0.3, -1.922], [0.4, -2.539], [0.5, -3.149],
        [0.6, -3.747]
    ]
    totU = np.array(totU)
    
    num = np.arange(100, 700, 100)
    up, uc = [], []
    
    for i, t in enumerate(totalEnergy_cpp):
        u1 = np.mean(np.array(t)) / num[i]
        u2 = np.mean(np.array(totalEnergy_py[i][1000:])) / num[i]
        uc.append(u1)
        up.append(u2)
    
    plt.figure(figsize=(8, 6))
    plt.plot(xx2, np.array(uc), color="r", linestyle="", marker="1", markersize=12, label="C++")
    plt.plot(xx2, np.array(up), color="g", linestyle="", marker="2", markersize=12, label="Python")
    plt.plot(totU[:,0], totU[:,1], color="b", linestyle="--", marker="", label="Reference")
    
    plt.xlabel("Density, " + r"$\rho \sigma^3$")
    plt.ylabel("Energy per particle, " + r"$U/N$")
    
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("energy_gca.png")
    plt.show()
    
    # Calculate and test chemical potential
    print("\n=== Chemical Potential Calculation ===")
    import MBWR_EOS
    
    test = {"temperature": 2.0, "sigma": 1., "component": 1, "epsilon": 1., "wave": 1.0}
    CP = []
    
    xx2 = np.arange(0.1, 0.7, 0.1)
    mbwr = MBWR_EOS.MBWR_EOS(test)
    for i in xx2:
        mbwr.rho = i
        chemP = mbwr.exChemicalPotential()
        CP.append(chemP[0][0])
    
    CP = np.array(CP)
    
    # data from CMC result of other program
    yy = np.array([-0.124232, -0.186052, -0.172991, -0.0584114, 0.244226, 0.771789])
    xx = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    
    tCPex_PY = [np.mean(np.array(i[2000:])) for i in cp_py]
    tCPex_PY = -np.log(tCPex_PY)
    
    tCPex_CPP = [np.mean(np.array(i)) for i in cp_cpp]
    tCPex_CPP = -np.log(tCPex_CPP)
    
    xx2 = np.arange(0.1, 0.7, 0.1)
    uts = []
    for i in xx2:
        ut = uTail(i, 2.5)
        uts.append(ut)
    
    tCPex_PY = tCPex_PY + (1./2.) * 2.0 * np.array(uts)  # unit: 1
    tCPex_CPP = tCPex_CPP + (1./2.) * 2.0 * np.array(uts)  # unit: 1
    
    plt.figure(figsize=(8, 6))
    plt.plot(testChemPex[:,0], testChemPex[:,1]/2., color="m", linestyle=":", marker="", label="Frankel")
    #plt.plot(xx, yy, color="b", linestyle="", marker="s", markersize=8, label="CMC")
    plt.plot(xx2, tCPex_PY, color="g", linestyle="", marker="1", markersize=12, label="Python")
    plt.plot(xx2, tCPex_CPP, color="r", linestyle="", marker="2", markersize=12, label="C++")
    plt.plot(xx2, CP, color="b", linestyle="", marker="o", markerfacecolor="none", markersize=8, label="MBWR")
    
    plt.xlabel("Density, " + r"$\rho \sigma^3$")
    plt.ylabel("Excess Chemical Potential, " + r"$\beta \mu^{ex}$")
    
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("chemical_potential_gca.png")
    plt.show()


if __name__ == "__main__":
    test_thermodynamics_metropolis()
    test_thermodynamics_gca()
