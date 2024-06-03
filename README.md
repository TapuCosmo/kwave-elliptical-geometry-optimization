# k-Wave Elliptical Geometry Optimization

This project uses parameter optimization to select the optimal transducer locations
to maximize the FSIM image quality metric measured between the ground truth and
time-reversal reconstructed image.

## Requirements
* An NVIDIA GPU with CUDA support
* Python 3.11

## Install
1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`

## Scripts
* `optimize.py`: Runs optimization for the parameters defined in the file.
* `run_simulation.py`: Plots the reconstruction for the parameters defined in the file.
* `get_optimized_parameters.py`: Extracts the parameters corresponding to the sample with the highest FSIM from the checkpoint.

## Notes
When running `optimize.py`, some of the trials will result in errors.
This is expected in order to force an early exit for parameters that
result in incorrect numbers of transducers being placed.

Fixed transducers are placed at the start and end of the arc.
When the number of transducers is odd, a third fixed transducer
is placed in the middle of the arc.