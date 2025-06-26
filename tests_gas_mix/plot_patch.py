import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Assuming the following utility functions exist
def get_angle_list(angles, length):
    return np.full(length, angles) if isinstance(angles, (int, float)) else np.array(angles)


def _rotate_dim2(v, angle):
    """ Rotate a 2D vector by a given angle. """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def plot_source_patches(node_coords, size, angles, **kwargs):
    # Get lines and polygons from the source_patches function
    lines, polys, _ = source_patches(node_coords, size, angles, **kwargs)

    # Create a plot
    fig, ax = plt.subplots()

    # Plot the lines
    for line in lines:
        ax.plot(*zip(*line), color='k')  # Use a color of your choice

    # Add each polygon (patch) to the plot
    for poly in polys:
        ax.add_patch(poly)

    # Set limits and labels
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_title('Source Patches')

    # Show the plot
    plt.show()


# Example usage
node_coords = np.array([[0, 0], [5, 5], [-5, -5]])  # Example coordinates
size = 1.0  # Size of the patches
angles = [0, np.pi / 4, -np.pi / 4]  # Angles in radians

# Call the plot function
plot_source_patches(node_coords, size, angles, patch_faceolor='silver', patch_edgecolor='black')