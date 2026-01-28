import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm

import sys
import time
import math

from PotentialType import PotentialType

# Import the CPP version MC
import build.MonteCarlo as MC

epsilon = 1e-9

class MonteCarlo():
    """
    This class is a Monte Carlo Simulation for Lennard-Jones fluids,
    in Canonical Ensemble.
    Temperature: kb * T / epsilon
    Size: sigma
    """

    def __init__(self, system):

        # 1. cells + rho
        # 2. num + size
        if "numberOfParticles" and "size" in system:
            self.box   = system["size"]
            self.num   = system["numberOfParticles"]
            self.cells = np.ceil( float( self.num / np.prod(self.box) )**(1./3.) * self.box - epsilon )
            self.cells = (self.cells).astype(np.int64)
            self.rho   = float(self.num) / np.prod(self.box)
        elif "cells" and "density" in system:
            self.cells = system["cells"]
            self.num   = np.prod(system["cells"])
            self.rho   = system["density"]
            self.box   = np.ones(3) * (self.num / self.rho)**(1./3.)
        else:
            raise ValueError("numberOfParticles or cells must be in system")

        self.rCut  = system["rCut"]
        self.drMax = system["drMax"]
        self.vol   = np.prod(self.box)

        # Temperature is equal to kb * T / epsilon
        self.temperature = system["temperature"]

        # Verbose flag for output control
        if "verbose" in system:
            self.verbose = system["verbose"]
        else:
            self.verbose = False

        self.r = np.zeros([self.num, 3])

        sr3 = 1.0 / self.rCut**3
        self.pressureDelta = np.pi * (8.0/3.0) * ( sr3**3 - sr3 ) * self.rho**2
        if self.verbose:
            print("pressure delta: " + str(self.pressureDelta) )
        self.pressureLrc = np.pi * ( (32./9.) * sr3**3 - (16./3.) * sr3 ) * self.rho**2
        if self.verbose:
            print("Pressure lrc: " + str(self.pressureLrc) )
        self.Ulrc = np.pi * ( (8./9.) * sr3**3 - (8./3.) * sr3 ) * self.rho
        if self.verbose:
            print("Internal Energy LRC: " + str(self.Ulrc) )

        # Algorithm selection: read from system["method"], default to "metropolis"
        if "method" in system:
            self.method = system["method"].lower()
        else:
            self.method = "metropolis"
        
        # Validate method
        if self.method not in ["metropolis", "gca"]:
            raise ValueError(f"Invalid method: {self.method}. Must be 'metropolis' or 'gca'")
        
        if self.verbose:
            print(f"Using algorithm: {self.method}")

        if ("Version" in system) and system["Version"] == "CPP":
            self.version = "CPP"
        else:
            self.version = "Python"

        if self.version == "CPP":

            dim = 3

            if "isNeighborList" in system:
                self.isNeighborList = system["isNeighborList"]
            else:
                self.isNeighborList = False

            if ("method" in system) and (self.method == "gca" ):
                self.isNeighborList = True

            self.mc = MC.MonteCarlo( self.num, dim, self.temperature, self.rCut, self.isNeighborList, self.verbose)
            (self.mc).SetBox( self.box )

            # Set the trajectory filename
            if ("Filename" in system):
                (self.mc).SetFileName( system["Filename"] )

            # Set the initial step
            if ("InitStep" in system):
                (self.mc).SetInitStep( system["InitStep"] )

            if ("method" in system) and (self.method == "gca"):
                (self.mc).SetMethod("GCA")

    @staticmethod
    def initCoords(pos, cells):
        """
        All particles are set at the center of a simple square lattice,
        at the initial state
        """

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

    @staticmethod
    def potential_1( ri, box, r_cut, r ):
        """
        Calculate the interaction between particle $i$ and other particles.
        The Lennard-Jones potential is used.
        """

        dim      = len(r[0])
        r_cut_sq = r_cut ** 2

        assert dim == 3, "Dimension error for r in potential_1"

        sr2_ovr = 1.77 # Overlap threshold (pot > 100)

        rij      = ri - r
        rij      = rij - np.rint(rij)  # Periodical boundary conditions
        rij      = rij * box
        rij2     = np.sum(rij**2, axis = 1)
        in_range = rij2 < r_cut_sq
        sr2      = np.where(in_range, 1./rij2, 0.)

        ovr = sr2 > sr2_ovr
        if np.any(ovr):
            partial = PotentialType( pot = 0.0, vir = 0.0, ovr = True )
            return partial

        sr6 = sr2 ** 3
        sr12 = sr6 ** 2

        pot = sr12 - sr6
        vir = pot + sr12
        partial = PotentialType( pot=np.sum(pot), vir=np.sum(vir), ovr=False )

        partial.pot = partial.pot * 4.0          # 4 * epsilon
        partial.vir = partial.vir * 24.0 / 3.0   # 24 * epsilon and divide by 3.0

        return partial

    @staticmethod
    def potential( box, r_cut, r ):
        """
        Calculate the total energy of the system
        """

        dim = len(r[0])
        num = len(r)
        assert dim == 3, "Dimension error for r in potential"

        total = PotentialType ( pot=0.0, vir=0.0, ovr=False )

        for i in range(num-1):
            partial = MonteCarlo.potential_1 ( r[i,:], box, r_cut, r[i+1:,:] )
            if partial.ovr:
                total.ovr = True
                break
            total = total + partial

        return total

    @staticmethod
    def random_translate_vector ( dr_max, old ):
        zeta = np.random.rand(3)
        zeta = 2.0*zeta - 1.0
        return old + zeta * dr_max

    @staticmethod
    def metropolis ( delta ):
        exponent_guard = 75.0

        if delta > exponent_guard: # Too high, reject without evaluating
            return False
        elif delta < 0.0:
            return True
        else:
            zeta = np.random.rand()
            return np.exp(-delta) > zeta

    # ============== Free Rejection Cluster Algorithm ================== #
    @staticmethod
    def compute_energy_change(pos_old, pos_new, other_positions, box, r_cut):
        '''
        Calculate the change in potential energy and virial when a particle
        moves from pos_old to pos_new.

        Parameters:
            pos_old: Old position of particle
            pos_new: New position of particle
            other_positions: Positions of other particles
            box: Box dimensions
            r_cut: Cutoff radius

        Returns:
            delta_potential: Change in potential energy for each other particle
            delta_virial: Change in virial for each other particle
        '''

        r_cut_sq = r_cut ** 2

        # Calculate the potential and virial between one particle and a group of particles.
        def _calc_potential_virial(pos, positions):
            # Calculate the distance vector and its squared length.
            rij = pos - positions
            rij = rij - np.rint(rij) # PBC
            rij = rij * box
            distance_sq = np.sum(rij**2, axis=1)

            # Apply cutoff: only consider pairs within cutoff distance
            mask = distance_sq < r_cut_sq

            # Initialize arrays to store the potential and virial component
            potential = np.zeros(len(positions))
            virial = np.zeros(len(positions))

            # Only calculate the potential and virial if the distance is less than the cutoff.
            if np.any(mask):

                valid_distances = distance_sq[mask]

                # Avoid division by zero
                safe_distances = np.maximum(valid_distances, 1e-10)

                # Calculate the LJ interaction
                inv2 = 1.0 / safe_distances
                inv6 = inv2 ** 3
                inv12 = inv6 ** 2

                # Calculate the potential and virial
                unscaled_potential = inv12 - inv6
                unscaled_virial = unscaled_potential + inv12

                potential[mask] = unscaled_potential * 4.0
                virial[mask] = unscaled_virial * 8.0

            return potential, virial

        # Calculate the old state
        potential_old, virial_old = _calc_potential_virial(pos_old, other_positions)
        
        # Calculate the new state
        potential_new, virial_new = _calc_potential_virial(pos_new, other_positions)

        # Return the difference in potential and virial
        delta_potential = potential_new - potential_old
        delta_virial = virial_new - virial_old

        return delta_potential, delta_virial

    def gca_step(self):
        """
        Perform one step of Geometric Cluster Algorithm (GCA).

        Returns:
            total_delta_potential: Total change in potential energy
            total_delta_virial: Total change in virial
            cluster_size: Size of cluster moved
        """

        pivot = np.random.rand(3) - 0.5
        seed_idx = np.random.randint(self.num)

        stack = [seed_idx]
        in_cluster = np.zeros(self.num, dtype=bool)
        in_cluster[seed_idx] = True

        # Record the system change
        total_delta_potential = 0.0
        total_delta_virial = 0.0

        cluster_size = 0
        r_cut_sq = self.rCut ** 2

        # Used to record the boundary energy generated by a particle "rejecting" other neighbors before deciding to join the cluster
        boundary_potential_accum = np.zeros(self.num)
        boundary_virial_accum = np.zeros(self.num)

        while stack:

            i = stack.pop()
            cluster_size += 1

            pos_old = self.r[i].copy()

            # Calculate the new position
            pos_new = 2.0 * pivot - pos_old
            pos_new = pos_new - np.rint(pos_new)

            # Renew the position of the particle
            self.r[i] = pos_new

            # Find the neighbor not in the cluster
            candidates_idx = np.where(~in_cluster)[0]
            if len(candidates_idx) == 0:
                continue

            candidates_positions = self.r[candidates_idx]

            # Calculate the energy and virial change
            delta_potential, delta_virial = self.compute_energy_change(pos_old, pos_new, candidates_positions, self.box, self.rCut)

            # Random acceptance probability
            prob = np.zeros_like(delta_potential)
            mask_pos = delta_potential > 0
            prob[mask_pos] = 1 - np.exp( -delta_potential[mask_pos] / self.temperature )

            random_vals = np.random.rand(len(candidates_idx))
            should_join = random_vals < prob

            # ==== Join the cluster ==== #
            if np.any(should_join):
                new_members = candidates_idx[should_join]

                # These particles have now decided to join, meaning they are no longer boundary particles.
                # When other neighbors visited them earlier (if they rejected), the energy generated was added to total_delta_potential.
                # Now that they are inside, those interactions become internal interactions (dE=0), so the previously added contributions must be undone.
                total_delta_potential -= np.sum(boundary_potential_accum[new_members])
                total_delta_virial -= np.sum(boundary_virial_accum[new_members])

                # Clear their accumulated values (to prevent duplicate subtraction, although logically they won't be accumulated again after joining the cluster)
                boundary_potential_accum[new_members] = 0.0
                boundary_virial_accum[new_members] = 0.0

                # Join the cluster
                in_cluster[new_members] = True
                stack.extend(new_members)

                # Note: For the pair i and new_members, they are now internal bonds with zero energy difference,
                # so there is no need to add delta_potential[should_join].

            # ====== Reject the cluster ====== #
            mask_stay = ~should_join
            if np.any(mask_stay):
                stay_indices = candidates_idx[mask_stay]

                # These particles remain outside, forming the cluster boundary.
                # Accumulate the actual boundary energy change
                potential_change = delta_potential[mask_stay]
                virial_change = delta_virial[mask_stay]

                total_delta_potential += np.sum(potential_change)
                total_delta_virial += np.sum(virial_change)

                # Remember those particles' boundary energy changes
                boundary_potential_accum[stay_indices] += potential_change
                boundary_virial_accum[stay_indices] += virial_change

        return total_delta_potential, total_delta_virial, cluster_size

    # ============ Free Rejection Cluster Algorithm, Done ============ #


    def MC_NVT( self, nBlock=10, nStep=1000, interval = 20 ):
        """
        Monte Carlo simulation in NVT ensemble.
        
        Parameters:
            nBlock: Number of blocks
            nStep: Number of steps per block
            interval: Trajectory output interval
        
        Returns:
            pressures: Pressure array
            moveRatios: Move ratio array
            total: Total energy array
            pots: Chemical potential array
            poss: Position snapshots
        """

        self.r = self.initCoords(self.r, self.cells)

        # Initial energy and check the particle position
        total = self.potential ( self.box, self.rCut, self.r )
        assert not total.ovr, "Overlap in the initial configuration"

        total.pot = total.pot + self.Ulrc * self.num

        moveRatio = 0.0
        moveRatios = []

        pressures   = []
        totalEnergy = []
        pots        = []

        poss = []

        for block in tqdm(range(nBlock)):

            for step in range(nStep):

                # ================== Algorithm Selection ================== #
                if self.method == "gca":
                    # GCA algorithm
                    delta_potential, delta_virial, cluster_size = self.gca_step()
                    total.pot += delta_potential
                    total.vir += delta_virial
                    moveRatio = cluster_size / self.num
                    
                else:  # metropolis
                    # Metropolis algorithm
                    moves = 0
                    for i in range(self.num):
                        rj = np.delete(self.r, i, 0) # Array of all the other particles.
                        partial_old = self.potential_1 ( self.r[i,:], self.box, self.rCut, rj )
                        assert not partial_old.ovr, "Overlap in the current configuration"

                        ri = self.random_translate_vector ( self.drMax/self.box, self.r[i,:] )
                        ri = ri - np.rint ( ri )

                        partial_new = self.potential_1 ( ri, self.box, self.rCut, rj )

                        if not partial_new.ovr:
                            delta = partial_new.pot - partial_old.pot
                            delta = delta / self.temperature

                            if self.metropolis ( delta ):
                                total  = total + partial_new - partial_old
                                self.r[i,:] = ri
                                moves  = moves + 1
                    
                    moveRatio = moves / self.num
                # ================== Algorithm Selection Done ================== #

                # Test Particle Method
                rj = np.random.rand(3) - 0.5
                testParticle = self.potential_1 ( rj, self.box, self.rCut, self.r )
                if testParticle.ovr:
                    pot = 0
                else:
                    pot = np.exp( - testParticle.pot / self.temperature )
                pots.append(pot)
                # End of Test Particle Method

                #pressure = rho * temperature + total.vir / vol + pressureDelta
                pressure = self.rho * self.temperature + total.vir / self.vol + self.pressureLrc

                if (block > 0) and ( step%interval == 0 ):
                    poss.append( (self.r).copy())

                # Adjust drMax (only for Metropolis)
                if self.method == "metropolis":
                    if moveRatio > 0.55:
                        self.drMax *= 1.05
                    elif moveRatio < 0.45:
                        self.drMax *= 0.95

                moveRatios.append(moveRatio)
                pressures.append(pressure)
                totalEnergy.append(total)

        total = self.potential ( self.box, self.rCut, self.r )
        assert not total.ovr, 'Overlap in final configuration'

        total = np.array( [it.pot for it in totalEnergy] )

        return pressures, moveRatios, total, pots, poss

    def MC_NVT_CPP( self, nBlock=10, nStep=1000, interval = 20 ):

        # Initialize energy and check particle positions
        self.mc.InitPosition()
        total = self.mc.GetPotential()
        assert not total["overlap"], 'Overlap in initial configuration'

        time_start = time.time()

        steps = int( nBlock * nStep )

        results = self.mc.MCrun(steps, self.drMax, interval)

        totalEnergy = results["potential"] + self.Ulrc * self.num
        pressures   = self.rho * self.temperature + results["vir"] / self.vol + self.pressureLrc
        moveRatios  = results["moveRatios"]

        time_end = time.time()
        if self.verbose:
            print("Running Time: " + str(time_end-time_start) + " s")

        cp = results["chemicalP"]

        chemP = np.mean( np.array( cp ) )
        chemP = - np.log( chemP ) + 2 * self.Ulrc / self.temperature
        # excess chemical potential, unit 1

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

        total = self.mc.GetPotential()
        assert not total["overlap"], 'Overlap in final configuration'

        return pressures, moveRatios, totalEnergy, cp, chemP
        # excess chemical potential unit in 1

    @staticmethod
    def rdf( pos, box, n, dr = 0.02, totalStep = 1000 ):
        """
        Calculate the radius distribution function
        Box must be a cube, in this version
        n: number of particles
        dr: in sigma units
        """

        dr   = dr / box                # in box = 1 units
        nk   = math.floor( 0.5 / dr )  # Accumulate out to half box length
        rMax = nk * dr                 # Actual rMax ( box = 1 units )

        h     = np.zeros( nk, dtype=np.int64 )  # Histogram bins initialized to zeros
        nStep = 0                               # Counts configurations


        while True:
            if (nStep >= totalStep) or (nStep >= len(pos)):
                break

            r = np.array( pos[nStep] )

            nShift = n//2
            for shift in range(nShift):
                rij         = r - np.roll(r, shift+1, axis=0)
                rij         = rij - np.rint(rij)
                rij_mag     = np.sqrt( np.sum(rij**2, axis=1) )
                hist, edges = np.histogram( rij_mag, bins=nk, range=(0.0, rMax) )
                factor      = 1 if n%2==0 and shift == nShift-1 else 2
                h           = h + factor* hist
            nStep = nStep + 1 # Increment configuration counter ready for next time

        h_id = ( 4.0 * np.pi * float(n) / 3.0 ) * ( edges[1:nk+1]**3 - edges[0:nk]**3 )
        g = h / h_id / ( n*nStep )

        edges = edges * box                           # Convert bin edges back to sigma = 1 units
        rMid  = 0.5 * ( edges[0:nk] + edges[1:nk+1] ) # Mid points of bins

        """
        1 2 3 4 5 6
        6 1 2 3 4 5 (1, 6) (1, 2)
        5 6 1 2 3 4 (1, 5) (1, 3)
        4 5 6 1 2 3 (1, 4) (1, 4)
        """

        """
        1 2 3 4 5
        5 1 2 3 4 (1, 5) (1, 2)
        4 5 1 2 3 (1, 4) (1, 3)
        """
        return rMid, g

    @staticmethod
    def readXYZTrajectory( filename ):
        """
        Read XYZ trajectory file
        """

        frames = []
        with open(filename, "r") as f:
            while True:
                numAtomsLine = f.readline()
                if not numAtomsLine:
                    break

                numAtoms = int(numAtomsLine.strip())
                commentLine = f.readline()

                currentFrame = []
                for _ in range(numAtoms):
                    atomLine = f.readline()
                    atomData = atomLine.split()
                    if not atomData:
                        raise ValueError("Unexpected end of file while reading atom data.")
                    x, y, z = map(float, atomData[-3:])
                    currentFrame.append([x, y, z])

                currentFrame = np.array(currentFrame)

                frames.append(currentFrame)

        return np.array(frames)

if __name__ == "__main__":

    system = {}
    system["size"]              = np.array( [10, 10, 10] )
    system["numberOfParticles"] = 400
    system["temperature"]       = 2.
    system["rCut"]              = 2.5
    system["drMax"]             = 0.25
    system["Version"]           = "CPP"
    system["isNeighborList"]    = True
    system["method"]            = "gca"

    nBlock = 10
    nStep  = 1000

    MC = MonteCarlo(system)
    pressure1, moveRatio1, totalEnergy1, pots1, _ = MC.MC_NVT(nBlock=nBlock*2, nStep=nStep)
    pressure,  moveRatio,  totalEnergy,  pots,  _ = MC.MC_NVT_CPP(nBlock=nBlock*4, nStep=nStep)

    # Pressure = 0.7062 (rho = 0.4)
    print( "CPP Pressure: " + str( np.mean( np.array(pressure[4000:] ) ) ) )
    print( "PY  Pressure: " + str( np.mean( np.array(pressure1[4000:]) ) ) )

    # Total Energy = -1014 (rho = 0.4, number of particles = 400)
    print( "CPP Total Energy: " + str( np.mean( np.array(totalEnergy[4000: ]) ) ) )
    print( "PY  Total Energy: " + str( np.mean( np.array(totalEnergy1[4000:]) ) ) )

    def uTail ( rho, rCut ):
        tail = (8./3.) * np.pi * rho * ( (1./3.) * (1./rCut)**9 - (1./rCut)**3 )
        return tail

    tCPex_PY = np.mean( np.array( pots1[2000:] ) )
    tCPex_PY = - np.log( tCPex_PY )

    tCPex_CPP = np.mean( np.array( pots ) )
    tCPex_CPP = - np.log( tCPex_CPP )

    ut = uTail ( 0.4, 2.5 )

    tCPex_PY  = tCPex_PY  + (1./2.) * 2.0 * ut # unit: 1
    tCPex_CPP = tCPex_CPP + (1./2.) * 2.0 * ut # unit: 1

    # Excess chemical potential = -0.4489 (MBWR result) (rho = 0.4, number of particles = 400)
    print( f"CPP Excess chemical potential: {tCPex_CPP}" )
    print( f"PY Excess chemical potential: {tCPex_PY}" )