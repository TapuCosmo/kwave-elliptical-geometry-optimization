import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import cv2
import math
import random
import builtins
import logging

# Override k-Wave files with patched files
def _import(name, *args, **kwargs):
  if name == "kwave.kWaveSimulation":
    name = "overrides.kWaveSimulationNew"
  elif name == "kwave.ksource":
    name = "overrides.kSourceNew"
  elif name == "kwave.executor":
    name = "overrides.executorNew"
  return original_import(name, *args, **kwargs)
original_import = builtins.__import__
builtins.__import__ = _import

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.utils.data import scale_SI
from kwave.utils.mapgen import make_cart_circle, make_circle
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.utils.filters import smooth
from kwave.utils.signals import add_noise
from kwave.utils.conversion import cart2grid
from kwave.utils.interp import interp_cart_data
from kwave.data import Vector

from scipy.ndimage import binary_dilation

from image_similarity_measures.quality_metrics import fsim

from generateP0 import generateP0
from trials import make_cart_symmetric_horizontal_ellipse
from PIL import Image, ImageDraw, ImageEnhance

# Set optimized parameters here
# Transducer and geometry configuration
# The eccentricity for both the target region and transducer array arc
eccentricity: float = 0.8
# The number of transducers to use
sensorPoints: int = 32
# The optimized parameters
# params = {"t0": <float>, "t1": <float>, ...}
params = {"t0": 0.8565546363017733, "t1": 0.7607931132612037, "t2": 0.01713155297173651, "t3": 0.7464929010602526, "t4": 0.9794597605298625, "t5": 0.721059606164659, "t6": 0.8301448312556592, "t7": 0.6963681470568974, "t8": 0.4401358968706735, "t9": 0.9243624017925046, "t10": 0.7942756237641654, "t11": 0.9953112483979484, "t12": 0.09198254078578515, "t13": 0.9089109150902258, "t14": 0.8705587212531747}
seed = 1

logging.basicConfig(level=logging.ERROR)

random.seed(seed)

# The size (in grid units) of the perfectly matched layer (PML)
pml_size = 25
# number of grid points in the x (row) direction
Nx: int = 512
# size of the domain in the x direction [m]
x: float = 0.08
# Should only be math.pi
# Other values will produce unexpected results
sensorAngle: float = math.pi # [rad]
# How far to position the transducers from the target region
sensor_offset: float = 0.01
# grid point spacing in the x direction [m]
dx: float = x / Nx
# sound speed [m/s]
sound_speed: float = 1500

# sensor position [m]
sensor_pos = np.array([0, 0])
# Convert width and eccentricity to bounding box size
ellipseWidth = sensor_offset + 350 * dx
ellipseHeight = ellipseWidth * math.sqrt(1 - eccentricity ** 2)
# startingConfig = {}
# for i in range(sensorPoints // 2 - 1):
#   startingConfig[f"t{i}"] = (i + 1) / (sensorPoints / 2) / (1 - 1 / sensorPoints)

# Create sensor mask based on params
sensor_positions = np.empty(15)
for i in range(15):
  # sensor_positions[i] = startingConfig[f"t{i}"]
  sensor_positions[i] = params[f"t{i}"]
cart_sensor_mask = make_cart_symmetric_horizontal_ellipse(sensor_pos, ellipseWidth, ellipseHeight, num_points=sensorPoints, max_angle=sensorAngle, sensor_positions=sensor_positions)

# Create p0 from random branching geometry
p0 = np.array(generateP0(seed=seed, imageSize=(1024, 1024), drawScale=500, ellipseWidth=700, eccentricity=eccentricity, thicknessScale=8, rotOffset=random.uniform(0, 360), posOffset=(random.randint(-100, 100), random.randint(-100, 100))).resize((Nx, Nx), resample=Image.Resampling.NEAREST), dtype=int) * 2
p0 = smooth(p0, True)

# medium
medium2 = kWaveMedium(sound_speed=1500)
# create the k-space grid
kgrid = kWaveGrid([Nx, Nx], [dx, dx])
kgrid.makeTime(medium2.sound_speed)

simulation_options = SimulationOptions(data_cast="single", smooth_p0=False, pml_inside=True, pml_size=pml_size, save_to_disk=True, pml_alpha=2)
execution_options = SimulationExecutionOptions(is_gpu_simulation=True, delete_data=True, verbose_level=0, show_sim_log=False)

# create instance of a sensor
sensor = kSensor()

# set sensor mask: the mask says at which points data should be recorded
sensor.mask = cart_sensor_mask
sensor.mask, _, _ = cart2grid(kgrid, sensor.mask, simulation_options.simulation_type.is_axisymmetric())

# make a source object
source = kSource()
source.p0 = p0

# run the simulation
sensor_data_2D = kspaceFirstOrder2D(
  medium=medium2,
  kgrid=kgrid,
  source=source,
  sensor=sensor,
  simulation_options=simulation_options,
  execution_options=execution_options
)
sensor_data_2D = sensor_data_2D["p"]

# Add 40 dB noise to sensor data before reconstruction
sensor_data_2D = add_noise(sensor_data_2D, 40)
# Transpose to prevent error (Python kWave seems to have a different input structure for reconstruction)
sensor_data_2D = sensor_data_2D.transpose()

# Set parameters for reconstruction
Nx_r = Nx + 2 * pml_size
dx_r: float = x / Nx_r

# Create new k-grid for reconstruction
kgrid_r = kWaveGrid([Nx_r, Nx_r], [dx_r, dx_r])
kgrid_r.setTime(kgrid.Nt, kgrid.dt)

# Reset source p0
source.p0 = np.array([])

# Set sensor data
sensor.time_reversal_boundary_data = sensor_data_2D
sensor.mask = cart_sensor_mask
sensor.mask, _, _ = cart2grid(kgrid_r, sensor.mask, simulation_options.simulation_type.is_axisymmetric())

# Run time reversal reconstruction
p0_r = kspaceFirstOrder2D(
  medium=medium2,
  kgrid=kgrid_r,
  source=source,
  sensor=sensor,
  simulation_options=simulation_options,
  execution_options=execution_options
)
p0_r = p0_r["p_final"]
# Remove PML from image
p0_r = p0_r[pml_size:(len(p0_r) - pml_size), pml_size:(len(p0_r[0]) - pml_size)]
# Transpose back to undo previous transpose
p0_r = p0_r.transpose()

# Create a mask to remove anything not inside the detection ellipse region for the input p0
# This shouldn't be required, but is here just in case the target exceeds the detector boundaries
p0RegionMask = Image.new("L", (Nx, Nx), 0)
p0RegionMaskDraw = ImageDraw.Draw(p0RegionMask)
p0RegionMaskDraw.ellipse(
  [
    (math.ceil((Nx - (ellipseWidth / dx)) / 2), math.ceil((Nx - (ellipseHeight / dx)) / 2)),
    (math.ceil((Nx + (ellipseWidth / dx)) / 2), math.ceil((Nx + (ellipseHeight / dx)) / 2))
  ],
  fill = 255,
  width = 0
)

# Create a mask to remove anything not inside the detection ellipse region for the reconstructed image
regionMask = Image.new("L", (Nx_r - 2 * pml_size, Nx_r - 2 * pml_size), 0)
regionMaskDraw = ImageDraw.Draw(regionMask)
regionMaskDraw.ellipse(
  [
    (math.ceil((Nx_r - 2 * pml_size - (ellipseWidth / dx_r)) / 2), math.ceil((Nx_r - 2 * pml_size - (ellipseHeight / dx_r)) / 2)),
    (math.ceil((Nx_r - 2 * pml_size + (ellipseWidth / dx_r)) / 2), math.ceil((Nx_r - 2 * pml_size + (ellipseHeight / dx_r)) / 2))
  ],
  fill = 255,
  width = 0
)

# Create plots
fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(p0, cmap="inferno")
ax[0, 0].set_title("Initial p0")

# Normalize the initial p0 and reconstructed data
p0 = (p0 - np.min(p0)) / (np.max(p0) - np.min(p0))
p0_r = (p0_r - np.min(p0_r)) / (np.max(p0_r) - np.min(p0_r))

# Plot with transducers
ax[0, 1].imshow(Image.composite(
  Image.fromarray(np.ones(p0_r.shape, dtype=np.uint8) * 255, "L"),
  Image.fromarray(np.uint8(cm.inferno(p0_r) * 255)),
  Image.fromarray(np.uint8(binary_dilation(sensor.mask[pml_size:(len(p0_r) + pml_size), pml_size:(len(p0_r[0]) + pml_size)], iterations=4) * 255), "L")
))
ax[0, 1].set_title(f"Reconstructed p0 ({sensorPoints} transducers)")

# Apply mask to p0
p0_masked = Image.composite(
  Image.fromarray(np.uint8(p0 * 255), "L"),
  Image.fromarray(np.ones(p0.shape, dtype=np.uint8) * 255, "L"),
  p0RegionMask.resize(p0.shape, resample=Image.Resampling.NEAREST)
)
# Scale normalized values to grayscale range
p0_r = p0_r * 255
# Apply filters and threshold
p0_r = cv2.edgePreservingFilter(p0_r, flags=1, sigma_s=20, sigma_r=0.2)
p0_r = cv2.adaptiveThreshold(p0_r, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 1)
# Apply mask
p0_r_masked = Image.composite(
  Image.fromarray(np.uint8(p0_r), "L"),
  Image.fromarray(np.ones(p0_r.shape, dtype=np.uint8) * 255, "L"),
  regionMask.resize(p0_r.shape, resample=Image.Resampling.NEAREST)
)
# Pad and rescale reconstructed image
paddedImage = Image.new("L", p0.shape, 255)
newWidth = Nx * Nx // Nx_r
paddedImage.paste(p0_r_masked.resize((newWidth, newWidth), resample=Image.Resampling.NEAREST), ((Nx - newWidth) // 2, (Nx - newWidth) // 2))
paddedImage = np.array(paddedImage)

# Calculate FSIM metric
recFSIM = fsim(np.expand_dims(np.array(p0_masked), axis=-1), np.expand_dims(np.array(paddedImage), axis=-1))

ax[1, 0].imshow(p0_masked, cmap="inferno")
ax[1, 0].set_title("Initial p0 (masked)")

ax[1, 1].imshow(paddedImage, cmap="inferno")
ax[1, 1].set_title(f"Reconstructed p0 (filtered + masked), FSIM = {recFSIM:.4f}")
plt.show()