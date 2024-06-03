import json
import os
from pathlib import Path
import math

# Parameters to locate checkpoint
eccentricity: float = 0.9
sensorPoints: int = 32
sensorAngle: float = math.pi

def parseTrialData(d):
  try:
    return json.loads(d[1])
  except:
    return None

checkpointPath = Path(f"./checkpoints/kwave-elliptical-geometry-optimization-{eccentricity:.2f}-{sensorPoints}-{sensorAngle:.3f}").resolve()

# Find the last experiment state
matchingFiles = []
for entry in os.scandir(checkpointPath):
  if entry.is_file() and entry.name.startswith("experiment_state") and entry.name.endswith(".json"):
    matchingFiles.append(entry.name)
matchingFiles.sort()

stateFilePath = Path(f"./checkpoints/kwave-elliptical-geometry-optimization-{eccentricity:.2f}-{sensorPoints}-{sensorAngle:.3f}/{matchingFiles[len(matchingFiles) - 1]}").resolve()

# Parse the experiment data to find the set of parameters with the highest average FSIM
data = json.load(open(stateFilePath))
trialData = list(map(parseTrialData, data["trial_data"]))
maxFSIM = max(map(lambda d: d["metric_analysis"]["avg_fsim"]["max"], filter(lambda d: d is not None and d["metric_analysis"], trialData)))
trialID = next(filter(lambda d: d is not None and d["metric_analysis"] and d["metric_analysis"]["avg_fsim"]["max"] == maxFSIM, trialData))["last_result"]["trial_id"]
params = json.loads(next(filter(lambda d: json.loads(d[0])["trial_id"] == trialID, data["trial_data"]))[0])["config"]

print(f"Max average FSIM: {maxFSIM}")
print(f"params = {json.dumps(params)}")