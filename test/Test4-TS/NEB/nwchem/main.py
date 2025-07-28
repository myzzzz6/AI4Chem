from fairchem.core import pretrained_mlip, FAIRChemCalculator
from ase.io import iread,read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase import Atoms
from ase.constraints import FixAtoms
import numpy as np
import os
from timeit import default_timer as timer
from torch import set_num_threads

set_num_threads(16)
#predictor = pretrained_mlip.get_predict_unit("uma-s-1", device="cpu")
predictor = pretrained_mlip.load_predict_unit(path="../../uma.pt", device="cpu")



if __name__ == "__main__":
    calc = FAIRChemCalculator(predictor, task_name="omol")
    atoms = read('2.xyz')  
    atoms.info.update({"spin": 1, "charge": 0})
    atoms.calc = calc
    start = timer()
    energy = atoms.get_total_energy()/units.eV
    print(energy)
    end = timer()
    atoms = read('2.xyz')  
    atoms.calc = calc
    print("time:",end - start)
