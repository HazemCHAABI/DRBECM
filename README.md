# DRBECM: Dynamic Role-Based Exploration with Connectivity Maintenance

This repository contains the Python implementation of the **DRBECM** algorithm, a distributed multi-robot exploration approach inspired by flocking behavior, described in our research paper:

> **Distributed Multi-Robot Exploration Approach With Connectivity Maintenance**  
> *Authors: Hazem CHaabi & Nathalie Mitton*  
> *[Link to the paper]*  

## Overview

The DRBECM algorithm dynamically assigns roles (**explorer** and **supporter**) to robots, balancing efficient exploration of unknown environments and maintaining reliable multi-hop communication with a base station. Explorers target unexplored frontiers, while supporters position themselves strategically as relay points to ensure connectivity, inspired by flocking behaviors found in nature.

## Repository Structure

The main file for the DRBECM algorithm:

- **`metrics_Flocking_inspired.py`**  
  *Implements the DRBECM algorithm described in the research paper.*

Additional files for alternative strategies and comparisons:

- `metrics_cheetah.py`
- `metrics_frontier_Exploration_and_sharing.py`
- `metrics_frontier_only_exploration_no_sharing.py`
- `metrics_RandomWalk.py`

## Setup & Requirements

### Prerequisites

- Python 3.x
- NumPy
- Matplotlib

Install required Python packages:

```bash
pip install numpy matplotlib
```

## How to Download and Run the DRBECM Code

### Step 1: Clone the Repository

```bash
git clone https://github.com/HazemCHAABI/DRBECM.git
cd DRBECM
```

### Step 2: Run the Simulation

To run the DRBECM simulation:

```bash
python metrics_Flocking_inspired.py
```

### Configuration Parameters

Adjust parameters in `metrics_Flocking_inspired.py`:

- `NUM_ROBOTS_LIST`: Number of robots (default: 5-15).
- `MAP_WIDTH`, `MAP_HEIGHT`: Environment size.
- `MAX_SPEED`: Robot speed.
- `COMMUNICATION_RANGE`: Communication distance.
- `SENSING_RANGE`: Sensor range.
- `NUM_RUNS`: Number of simulation runs.
- `MAX_FRAMES`: Iterations per simulation.

## Visualization

Enable real-time visualization by setting `VISUALIZE` to `True` in the script.

## Output Metrics

Metrics are saved in CSV format and include:

- Exploration time
- Coverage percentage
- Redundant exploration cells
- Total distance traveled

## Citation

If you use this work, please cite:

```bibtex
@INPROCEEDINGS{DRBECM,
  author={Chaabi, Hazem and Mitton, Nathalie},
  booktitle={2025 23rd International Symposium on Modeling and Optimization in Mobile, Ad Hoc, and Wireless Networks (WiOpt)}, 
  title={Distributed Multi-Robot Exploration Approach With Connectivity Maintenance}, 
  year={2025}}
```



## Contact

For questions, please contact hazem.chaabi@inria.fr

