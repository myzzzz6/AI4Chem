from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS, MDMin
from ase.mep import NEB
from ase.io import Trajectory
from ase.io import read,write
import os
from ase.units import Hartree,eV
from torch import set_num_threads
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import numpy as np

predictor = pretrained_mlip.load_predict_unit(path="/home/xchen/Test/metaUMA/uma.pt", device="cpu")
set_num_threads(16)

def plot_neb_energy(images, filename="neb_energy_profile.png"):
    images[0].calc = FAIRChemCalculator(predictor, task_name="omol")
    images[-1].calc = FAIRChemCalculator(predictor, task_name="omol")

    energies = np.array([atoms.get_total_energy()/eV for atoms in images])  # Convert to eV
    energies = energies - energies[0]
    num_images = len(images)

    distances = []
    cumulative_distance = 0.0
    if num_images > 1:
        for i in range(num_images - 1):
            cumulative_distance += np.linalg.norm(images[i+1].get_positions() - images[i].get_positions())
            distances.append(cumulative_distance)
        distances.insert(0, 0.0)
    else:
        distances = [0.0]

    plt.figure(figsize=(10, 6))
    plt.plot(distances, energies, marker='o', linestyle='-')
    plt.xlabel('Distance along Reaction Path (arbitrary units or RMSD)')
    plt.ylabel('Energy (eV)')
    plt.title('NEB Energy Profile')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)  #  
    plt.close()   


def optimize_structure(atoms, fmax=0.05, trajectory=None):
    atoms.info.update({"spin": 1, "charge": 0})
    atoms.calc = FAIRChemCalculator(predictor, task_name="omol")
    dyn = BFGS(atoms, trajectory=trajectory)
    dyn.run(fmax=fmax)
    return atoms.copy()

def run_neb(reactant, product, n_images=5, fmax=0.05, neb_traj='neb.traj'):
    # Step 1: Optimize reactant and product
    print("Optimizing reactant...")
    reactant_opt = optimize_structure(reactant, fmax=0.05, trajectory='reactant_opt.traj')

    print("Optimizing product...")
    product_opt = optimize_structure(product, fmax=0.05, trajectory='product_opt.traj')

    # Step 2: Generate NEB images
    print("Setting up NEB...")
    # Read initial images from neb.traj

    images = [reactant_opt]
    traj = Trajectory('neb.traj')
    imagesinter = [atoms for atoms in traj]
    images = [atoms for atoms in traj][-50:-2]
    images.insert(0, reactant_opt)
    images += [product_opt]
    neb = NEB(images)
    # Interpolate linearly the potisions of the three middle images:
    neb.interpolate()

    neb = NEB(images,climb=True)
    # No need to interpolate if reading from trajectory

    for image in images[1:-1]:
        image.info.update({"spin": 1, "charge": 0})
        image.calc = FAIRChemCalculator(predictor, task_name="omol")

    # Step 3: Run NEB optimization
    print("Running NEB optimization...")
    optimizer = MDMin(neb, trajectory=neb_traj, dt=0.05)
    optimizer.run(fmax=fmax)

    print("NEB calculation finished.")
    return images


r = read('R.xyz')
p = read('P.xyz')
images = run_neb(r, p, n_images=49, fmax=0.05)
plot_neb_energy(images, filename="singlet_neb_plot.png")
