"""
Test equivalence between Python and C++ implementations of Monte Carlo simulation.

This test verifies that the Python and C++ implementations produce
consistent results for the same simulation parameters.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))

import numpy as np

try:
    import MonteCarlo as MonteCarlo_py
    from PotentialType import PotentialType
except ImportError:
    print("Python MonteCarlo module not available, skipping test")
    sys.exit(0)

try:
    import build.MonteCarlo as MonteCarlo_cpp
except ImportError:
    print("C++ MonteCarlo module not available, skipping test")
    sys.exit(0)


def test_consistency():
    """Test equivalence between Python and C++ implementations."""
    # Configuration
    box = np.array([10, 10, 10])
    num = 200
    temperature = 2.0
    rCut = 2.5
    drMax = 0.25
    
    # Initialize particle positions
    cells = np.ceil(float(num / np.prod(box))**(1./3.) * box)
    cells = cells.astype(np.int64)
    r = np.zeros([num, 3])
    r = MonteCarlo_py.MonteCarlo.initCoords(r, cells)
    
    # Check initial energy
    total = MonteCarlo_py.MonteCarlo.potential(box, rCut, r)
    assert not total.ovr, "Overlap in initial configuration"
    
    # C++ version
    mc_cpp = MonteCarlo_cpp.MonteCarlo(num, 3, temperature, rCut, True, False)
    mc_cpp.SetBox(box)
    mc_cpp.SetPosition(r)
    total1 = mc_cpp.GetPotential()
    
    print("CPP total potential:", total1["pot"])
    print("Py  total potential:", total.pot)
    print("CPP total Virial:", total1["vir"])
    print("Py  total Virial:", total.vir)
    
    # Test displacement
    results = mc_cpp.MoveParticles(drMax)
    total2 = PotentialType(pot=results["pot"], vir=results["vir"], ovr=False) + total
    
    total1 = mc_cpp.GetPotential()
    r1 = mc_cpp.GetPosition()
    total = MonteCarlo_py.MonteCarlo.potential(box, rCut, r1)
    
    print("********* after displacement particle *********")
    print("CPP total potential:", total1["pot"])
    print("CPP total potential 2:", total2.pot)
    print("Py  total potential:", total.pot)
    print("CPP total Virial:", total1["vir"])
    print("CPP total Virial 2:", total2.vir)
    print("Py total Virial:", total.vir)
    
    # Compare results
    print("\n=== Consistency Test Results ===")
    print("Potential difference:", abs(total1["pot"] - total.pot))
    print("Virial difference:", abs(total1["vir"] - total.vir))
    
    # Assert consistency
    assert abs(total1["pot"] - total.pot) / total.pot < 1e-4, "Potential energy difference too large"
    assert abs(total1["vir"] - total.vir) / total.pot < 1e-4, "Virial difference too large"


if __name__ == "__main__":
    test_consistency()
