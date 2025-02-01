import build.MonteCarlo as MC

import numpy as np
import time
from itertools import product

import sys

def initCoords(pos, cells):
    num = len(pos)
    dim = len(pos[0])
    c   = np.zeros(dim)
    gap = 1. / cells

    assert dim == 3, "Wrong dimension, initCoords"

    n = 0
    for i, j, k in product(range(cells[0]), range(cells[1]), range(cells[2]) ):

        c      = np.array([i, j, k]) + 0.5
        c      = c * gap
        c      = c - 0.5
        pos[n] = c
        n      = n + 1

        if n >= num:
            return pos

    return pos

def MC_NVT_CPP(system, isNeighborList = True):

    box = system["size"]
    vol = np.prod(system["size"])
    num = system["numberOfParticles"]
    rho = float(num) / vol

    rCut  = system["rCut"]
    drMax = system["drMax"]
    cells = system["cells"]

    nBlock = system["numberOfBlocks"]
    nStep  = system["numberOfSteps"]

    temperature = system["temperature"]

    r = np.zeros([num, 3])
    r = initCoords(r, cells)

    sr3 = 1.0 / rCut**3
    pressureDelta = np.pi * (8.0/3.0) * ( sr3**3 - sr3 ) * rho**2
    print("pressure delta: " + str(pressureDelta) )
    pressureLrc = np.pi * ( (32./9.) * sr3**3 - (16./3.) * sr3 ) * rho**2
    print("Pressure lrc: " + str(pressureLrc) )
    Ulrc = np.pi * ( (8./9.) * sr3**3 - (8./3.) * sr3 ) * rho
    print("Internal Energy LRC: " + str(Ulrc) )

    mc = MC.MonteCarlo( num, 3, temperature, rCut, isNeighborList )
    mc.SetBox( box )

    # Initial energy and check the particle position

    mc.SetPosition( r )
    # mc.InitPosition()
    total1 = mc.GetPotential()
    assert not total1["overlap"], "Overlap in the initial configuration"

    """
    total = PotentialType ( pot=total1["pot"], vir=total1["vir"], ovr= (True if total1["overlap"] else False) )
    total.pot = total.pot + Ulrc * num
    """

    time_start = time.time()

    steps = int( nBlock * nStep )

    results = mc.MCrun(steps, drMax)

    totalEnergy = results["potential"] + Ulrc * num
    pressures   = rho * temperature + results["vir"] / vol + pressureLrc
    moveRatios  = results["moveRatios"]

    time_end = time.time()
    print("Running Time: " + str(time_end-time_start) + " s")

    pots = []

    '''
    moveRatios = []
    pressures   = []
    totalEnergy = []
    pots        = []

    moveRatio = 0.0

    for block in tqdm(range(nBlock)):

        for step in range(nStep):

            results = mc.MoveParticles( drMax )
            total  += PotentialType ( pot=results["pot"], vir=results["vir"], ovr= False )

            moves = results["moves"]

            # Test Particle Method
            pot = mc.TestParticles()
            pot = np.exp( - pot / temperature )
            pots.append(pot)
            # End of Test Particle Method

            moveRatio = moves / num
            #pressure = rho * temperature + total.vir / vol + pressureDelta
            pressure = rho * temperature + total.vir / vol + pressureLrc

            # Test ratio of accepted to attempted moves
            if moveRatio > 0.55:
                drMax *= 1.05
            elif moveRatio < 0.45:
                drMax *= 0.95

            moveRatios.append(moveRatio)
            pressures.append(pressure)
            totalEnergy.append(total)
    '''

    total = mc.GetPotential()
    assert not total["overlap"], 'Overlap in final configuration'

    return pressures, moveRatios, totalEnergy, pots


epsilon = 1e-9

system = {}
system["size"]              = np.array([10, 10, 10])
system["numberOfParticles"] = 27
system["temperature"]       = 2.
system["rCut"]              = 2.5
system["cells"]             = np.ceil( float( system["numberOfParticles"] / np.prod(system["size"]) )**(1./3.) * system["size"] - epsilon )
system["cells"]             = system["cells"].astype(np.int64)
system["drMax"]             = 0.25
system["numberOfBlocks"]    = 10
system["numberOfSteps"]     = 1000

pressure, moveRatio, totalEnergy, pots = MC_NVT_CPP(system, isNeighborList=True)

print( "CPP Pressure: " + str( np.mean( np.array(pressure[4000:] ) ) ) )
print( "CPP Total Energy: " + str( np.mean( np.array(totalEnergy[4000: ]) ) ) )

# num = 10
# temperature = 1.0
# rCut = 2.5
# box = [10.0, 10.0, 10.0]

# mc = MC.MonteCarlo(num, 3, temperature, rCut, True)
# mc.SetBox(box)

# r = np.array( [[8.,8.,8.], [9.,9.,9.],[5.5,8,8],[6.5,9,9],[1,8,8],[2,9,9],[8,8,1],[8,8,2],[3,3,3],[4,4,4]] )
# r = r / box - 0.5
# mc.SetPosition(r)
