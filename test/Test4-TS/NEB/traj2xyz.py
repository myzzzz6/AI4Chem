from ase.io import Trajectory, write

# Read the NEB trajectory
try:
    traj = Trajectory('neb.traj')
except FileNotFoundError:
    print("Error: Trajectory file 'neb.traj' not found.")
    exit()

# Select the last 50 images (excluding the last one)
images_to_write = [atoms for atoms in traj][-53:]

# Write the selected images to an xyz file
write('last_images.xyz', images_to_write)
