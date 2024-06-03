import numpy as np
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

from image_similarity_measures.quality_metrics import fsim

from generateP0 import generateP0
from PIL import Image, ImageDraw

# Run a trial on a random branch geometry target and return the FSIM
def runTrial(seed=0, pml_size=25, Nx=512, x=0.08, sensor_points=64, sensor_positions=None, sensor_angle=math.pi, sensor_offset=0.01, eccentricity=0, flip=False):
  logging.basicConfig(level=logging.ERROR)

  if len(sensor_positions) != sensor_points // 2 - 1:
    raise ValueError("len(sensor_positions) == sensor_points // 2 - 1 not satisfied")

  if seed is not None:
    random.seed(seed)

  # grid point spacing in the x direction [m]
  dx: float = x / Nx
  # sound speed [m/s]
  sound_speed: float = 1500
  # sensor position [m]
  sensor_pos = np.array([0, 0])
  # Convert width and eccentricity to bounding box size
  ellipseWidth = sensor_offset + 350 * dx
  ellipseHeight = ellipseWidth * math.sqrt(1 - eccentricity ** 2)

  simulation_options = SimulationOptions(data_cast="single", smooth_p0=False, pml_inside=True, pml_size=pml_size, save_to_disk=True, pml_alpha=2)
  execution_options = SimulationExecutionOptions(is_gpu_simulation=True, delete_data=True, verbose_level=0, show_sim_log=False)

  # Generate elliptical transducer geometry
  cart_sensor_mask = make_cart_symmetric_horizontal_ellipse(sensor_pos, ellipseWidth, ellipseHeight, num_points=sensor_points, max_angle=sensor_angle, sensor_positions=sensor_positions)

  # Generate random branch geometry as imaging target
  p0 = np.array(generateP0(
    imageSize=(1024, 1024),
    ellipseWidth=700,
    eccentricity=eccentricity,
    thicknessScale=8,
    rotOffset=random.uniform(0, 360) + (180 if flip else 0),
    posOffset=(random.randint(-100, 100), random.randint(-100, 100))
  ).resize((Nx, Nx), resample=Image.Resampling.NEAREST), dtype=int) * 2
  p0 = smooth(p0, True)

  # medium
  medium = kWaveMedium(sound_speed=1500)
  # create the k-space grid
  kgrid = kWaveGrid([Nx, Nx], [dx, dx])
  # Automatically create time range and step
  kgrid.makeTime(medium.sound_speed)

  # create instance of a sensor
  sensor = kSensor()

  # set sensor mask: the mask says at which points data should be recorded
  sensor.mask = cart_sensor_mask
  sensor.mask, _, _ = cart2grid(kgrid, sensor.mask, simulation_options.simulation_type.is_axisymmetric())

  # check if there are any transducers that were combined into one
  if np.count_nonzero(sensor.mask) != cart_sensor_mask.shape[1] or cart_sensor_mask.shape[1] != sensor_points:
    raise Exception("Overlapping transducers placed")

  # make a source object
  source = kSource()
  source.p0 = p0

  # Run the k-wave simulation
  sensor_data_2D = kspaceFirstOrder2D(
    medium=medium,
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
    medium=medium,
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

  # Normalize the initial p0 and reconstructed data
  p0 = (p0 - np.min(p0)) / (np.max(p0) - np.min(p0))
  p0_r = (p0_r - np.min(p0_r)) / (np.max(p0_r) - np.min(p0_r))

  # Apply mask to the initial image
  p0_masked = Image.composite(
    Image.fromarray(np.uint8(p0 * 255), "L"),
    Image.fromarray(np.ones(p0.shape, dtype=np.uint8) * 255, "L"),
    p0RegionMask.resize(p0.shape, resample=Image.Resampling.NEAREST)
  )
  # Scale reconstructed p0 to [0, 255]
  p0_r = p0_r * 255
  # Apply a type of low-pass filter with edge-preservation to the reconstructed image
  p0_r = cv2.edgePreservingFilter(p0_r, flags=1, sigma_s=20, sigma_r=0.2)
  # Apply thresholding
  p0_r = cv2.adaptiveThreshold(p0_r, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 1)
  # Apply the mask to the reconstructed image
  p0_r_masked = Image.composite(
    Image.fromarray(np.uint8(p0_r), "L"),
    Image.fromarray(np.ones(p0_r.shape, dtype=np.uint8) * 255, "L"),
    regionMask.resize(p0_r.shape, resample=Image.Resampling.NEAREST)
  )
  # Rescale and pad the reconstructed image so it has the same dimensions and pixels per distance as the initial image
  paddedImage = Image.new("L", p0.shape, 255)
  newWidth = Nx * Nx // Nx_r
  paddedImage.paste(p0_r_masked.resize((newWidth, newWidth), resample=Image.Resampling.NEAREST), ((Nx - newWidth) // 2, (Nx - newWidth) // 2))
  paddedImage = np.array(paddedImage)

  # Compute the FSIM image quality metric
  recFSIM = fsim(np.expand_dims(np.array(p0_masked), axis=-1), np.expand_dims(np.array(paddedImage), axis=-1))
  return recFSIM

# Utility function to create symmetric elliptical transducer geometry
def make_cart_symmetric_horizontal_ellipse(
  center=(0, 0),
  width=0.1,
  height=0.1,
  num_points=128,
  max_angle=math.pi,
  sensor_positions=np.array([])
):
  # Check input values
  if width <= 0:
    raise ValueError("The width must be positive.")
  if height <= 0:
    raise ValueError("The height must be positive.")

  t = np.empty(num_points, float)
  # Place points at ends
  t[0] = 0
  t[num_points - 1] = max_angle
  # Place point at center for odd number of points
  if num_points % 2 == 1:
    t[num_points // 2] = max_angle / 2
  # Place points symmetrically
  for i in range(num_points // 2 - 1):
    # t[i + 1] = t[i] + sensor_positions[i]
    # t[num_points - i - 2] = t[num_points - i - 1] - sensor_positions[i]
    t[i + 1] = sensor_positions[i] * max_angle / 2
    t[num_points - i - 2] = max_angle - sensor_positions[i] * max_angle / 2
  
  arc = np.array([-height * np.sin(t) / 2 + center[1], width * np.cos(t) / 2 + center[0]])

  return arc