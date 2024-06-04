import numpy as np
import math
import random
import logging
from pathlib import Path

from ray import train, tune
from ray.train import Checkpoint
from ray.tune.search.hyperopt import HyperOptSearch

from trials import runTrial

# Optimization configuration
# How many random targets to use for each sample
trialsPerSample: int = 10
# The max number of samples to run
# Actual number will be less due to errors from randomly selected parameters
# sometimes leading to overlapping transducers
numSamples: int = 1000

# Transducer and geometry configuration
# The eccentricity for both the target region and transducer array arc
eccentricity: float = 0.9
# The number of transducers to use
sensorPoints: int = 32

# The size of the signal acquisition simulation grid
Nx: int = 512
# The simulated size of the grid (square)
x: float = 0.08 # [m]
# Should only be math.pi
# Other values will produce unexpected results
sensorAngle: float = math.pi # [rad]
# How far to position the transducers from the target region
sensorOffset: float = 0.01 # [m]

logging.basicConfig(level=logging.ERROR)

checkpointPath = Path("./checkpoints").resolve()

# Objective function to maximize
def objective(sensorPositions):
  fsims = np.zeros(trialsPerSample * 2)
  # Run the trials for a single set of parameters
  for i in range(0, trialsPerSample * 2, 2):
    fsims[i] = runTrial(
      seed=i,
      eccentricity=eccentricity,
      Nx=Nx,
      x=x,
      sensor_points=sensorPoints,
      sensor_angle=sensorAngle,
      sensor_offset=sensorOffset,
      sensor_positions=sensorPositions
    )
    # Re-run with flipped target to reduce random orientation-related biases
    fsims[i + 1] = runTrial(
      seed=i,
      eccentricity=eccentricity,
      Nx=Nx,
      x=x,
      sensor_points=sensorPoints,
      sensor_angle=sensorAngle,
      sensor_offset=sensorOffset,
      sensor_positions=sensorPositions,
      flip=True
    )
    print(f"trial {i // 2 + 1}/{trialsPerSample} FSIM_0 = {fsims[i]:.4f} FSIM_1 = {fsims[i + 1]:.4f}")
  # Return the average FSIM across all trials as the objective function value
  avgFSIM = np.sum(fsims) / trialsPerSample / 2
  print(f"avg FSIM = {avgFSIM:.4f}")
  return avgFSIM

class OptimizeGeometry(tune.Trainable):
  def setup(self, config: dict):
    # Assign sensor positions according to chosen parameters
    self.sensorPositions = np.empty(sensorPoints // 2 - 1, float)
    for i in range(sensorPoints // 2 - 1):
      self.sensorPositions[i] = config[f"t{i}"]
  
  def step(self):
    # Evaluate the objective function and return the average FSIM
    fsim = objective(self.sensorPositions)
    return {"avg_fsim": fsim}

config = {}
startingConfig = {}
for i in range(sensorPoints // 2 - 1):
  config[f"t{i}"] = tune.uniform(0, 1)
  # Evenly space (by angle) the initial points
  startingConfig[f"t{i}"] = (i + 1) / (sensorPoints / 2) / (1 - 1 / sensorPoints)

algo = HyperOptSearch(points_to_evaluate=[startingConfig])
# Uncomment below if restoring checkpoint
# algo.restore(Path(f"./checkpoints/kwave-elliptical-geometry-optimization-{eccentricity:.2f}-{sensorPoints}-{sensorAngle:.3f}/searcher-state-2024-06-01_11-20-55.pkl").resolve())
tuner = tune.Tuner(
  tune.with_resources(OptimizeGeometry, resources={"gpu": 1}),
  tune_config=tune.TuneConfig(
    search_alg=algo,
    max_concurrent_trials=1,
    metric="avg_fsim",
    mode="max",
    num_samples=numSamples
  ),
  run_config=train.RunConfig(
    name=f"kwave-elliptical-geometry-optimization-{eccentricity:.2f}-{sensorPoints}-{sensorAngle:.3f}",
    storage_path=checkpointPath,
    stop={"training_iteration": 1},
    checkpoint_config=train.CheckpointConfig(
      checkpoint_at_end=False
    )
  ),
  # Comment out below if restoring checkpoint
  param_space=config
)
tuner.fit()