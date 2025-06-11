from ase.io import iread,read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase import Atoms
from ase.constraints import FixAtoms
import numpy as np
import os
import time
from fairchem.core import pretrained_mlip, FAIRChemCalculator
predictor = pretrained_mlip.get_predict_unit("uma-sm", device="cuda")

def UMA_streching(iniMol,top,bottom,nstep):
    # Set up a molecule
    molecule = iniMol
    molecule.calc = FAIRChemCalculator(predictor, task_name="omol")


    # Set the momenta corresponding to T=300K
    MaxwellBoltzmannDistribution(molecule, temperature_K=300)

    print("temperature:",molecule.get_temperature())

    # We want to run MD with constant energy using the VelocityVerlet algorithm.
    dyn = VelocityVerlet(molecule, 1 * units.fs,  trajectory='UMA-stratching.traj')  # 5 fs time step.
    istep0 = 0
    def printenergy(a=molecule,d=dyn):  # store a reference to atoms in the definition.
        """Function to print the potential, kinetic and total energy."""
        epot = a.get_potential_energy() / len(a)
        ekin = a.get_kinetic_energy() / len(a)
        del a.constraints
        positions = a.get_positions()
        for i in top:
            positions[i, 2] -= 0.0015 # Increment z-coordinate for top atoms

        for i in bottom:
            positions[i, 2] += 0.0015 # Decrement z-coordinate for bottom atoms
        a.set_positions(positions)
        c = FixAtoms(indices=top+bottom)
        a.set_constraint(c)
        print("STEP:",d.nsteps,'Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
              'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin) )

    # Now run the dynamics
    dyn.attach(printenergy, interval=1)
    printenergy()
    dyn.run(nstep)


if __name__ == "__main__":
    ini_struct = read('CNT.xyz')
    ini_struct.set_pbc((True, True,True))
    ini_struct.set_cell([[30,0,0],[0,30,0],[0,0,50]])
    pos_ini = np.array(ini_struct.get_positions())
    pos_ini[:,0] = pos_ini[:,0]+15
    pos_ini[:,1] = pos_ini[:,1]+15
    pos_ini[:,2] = pos_ini[:,2]+25
    ini_struct.set_positions(pos_ini)

    UMA_streching(ini_struct ,top=[0,1,4,5,8,9],bottom=[74,75,78,79,82,83],nstep =6000)
