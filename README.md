# SIC: Swarm Inspired Controllers

A Python-based simulation framework for studying swarm-inspired control strategies for shape manipulation and self-assembly using distributed robotic tiles.

## Overview

Swarm Inspired Controllers (SIC) is a research project that implements and compares different swarm-inspired control algorithms for coordinating distributed robotic tiles to manipulate objects. The system simulates a grid of intelligent tiles that can communicate with neighbors and collectively control the movement and rotation of objects placed on the grid.

## Core Concepts

### Distributed Tile System
- **Tiles**: Individual robotic units arranged in a grid that can sense, communicate, and act
- **Neighborhood Communication**: Each tile shares information with adjacent tiles
- **Collective Behavior**: Tiles work together to achieve global objectives through local interactions

### Shape Manipulation
- **Tetromino Objects**: Standard tetromino shapes (I, O, T, J, L, S, Z) that can be moved and rotated
- **Target Shapes**: Desired final configurations that tiles work to achieve
- **Contact Sensing**: Tiles detect when they are in contact with objects
- **Coverage Metrics**: Performance measured by how well tiles cover target shapes

## Architecture

### Main Components

#### 1. Environment (`Environment/`)
- **`Simulator.py`**: Main simulation engine that orchestrates the entire system
- **`Board.py`**: Grid management and tile coordination
- **`Tile.py`**: Individual tile implementation with sensing, communication, and behavior execution
- **`Tetromino.py`**: Tetromino shape representation and manipulation
- **`DataHandler.py`**: Data collection and storage utilities

#### 2. Control Behaviors (`Behaviors.py`)
Five different swarm-inspired control strategies:

1. **InfDiff (Information Diffusion)**: Traditional information diffusion approach
2. **Discrete**: Discrete state-based control with excitation signals
3. **Logistic**: Continuous control using logistic functions for symmetry breaking
4. **Gaussian**: Gaussian-based control for smooth transitions
5. **Fourier**: Fourier series-based control for complex behaviors

#### 3. Optimization (`Optimization.py`, `EvolutionarStrategies.py`)
- **Evolutionary Strategies**: CMA-ES, SimpleGA, OpenES, PEPG algorithms
- **Parameter Optimization**: Automated tuning of behavior parameters
- **Fitness Functions**: Performance evaluation

### Key Features

#### Simulation Capabilities
- **Real-time Visualization**: Pygame-based interactive simulation
- **Data Collection**: Comprehensive logging of system states and performance metrics
- **Animation Export**: GIF generation for analysis and presentation
- **Fault Tolerance**: Simulation of tile failures and system robustness

#### Experimental Framework
- **Comparative Studies**: Systematic comparison of different control strategies
- **Convergence Analysis**: Study of how systems evolve over time
- **Failure Analysis**: Investigation of when and why systems fail
- **Fault Tolerance Testing**: Performance under various failure scenarios

## Experiments

### 1. Extended Comparison (`Exp_Comparison.py`)
- Compares all five control behaviors across 500 runs
- Tests with all tetromino shapes and various initial conditions
- Measures position error, angle error, coverage, and operation time

### 2. Convergence Analysis (`Exp_Convergence.py`)
- Detailed study of how systems converge to solutions
- Time-series analysis of system evolution
- Visualization of convergence snapshots

### 3. Failure Analysis (`Exp_Fail_Analysis.py`)
- Studies angular error patterns and failure modes

### 4. Fault Tolerance (`Exp_FaultTolerance.py`)
- Tests system robustness under tile failures
- Measures performance degradation with increasing failure rates
- Evaluates which control strategies are most resilient

## Usage

### Basic Simulation
```python
from Environment.Simulator import Simulator
from Behaviors import Behaviors
from TunableParameters import TunableParameters

# Set up simulation parameters
setup = {
    'N': 20,                    # Grid size
    'TILE_SIZE': 20,           # Tile size in pixels
    'symbol': 'T',             # Tetromino shape
    'visualize': True,         # Enable visualization
    'max_iterations': 500      # Maximum simulation steps
}

# Configure behavior and parameters
TunableParameters.set_params()
Tile.execute_behavior = Behaviors.Logistic

# Run simulation
simulator = Simulator(setup)
simulator.run_simulation()
```

### Running Experiments
```bash
# Run comparison experiment
python Exp_Comparison.py

# Run convergence analysis
python Exp_Convergence.py

# Run failure analysis
python Exp_Fail_Analysis.py

# Run fault tolerance test
python Exp_FaultTolerance.py
```

## Dependencies

- **numpy**: Numerical computations
- **pygame**: Visualization and interaction
- **matplotlib**: Plotting and analysis
- **shapely**: Geometric operations
- **pandas**: Data manipulation
- **imageio**: Animation export
- **tqdm**: Progress tracking
- **cma**: CMA-ES optimization

## Project Structure
```
├── main.py # Main entry point
├── Behaviors.py # Control behavior implementations
├── TunableParameters.py # Parameter management
├── Optimization.py # Parameter optimization
├── EvolutionarStrategies.py # Evolutionary algorithms
├── Environment/ # Core simulation environment
│ ├── Simulator.py # Main simulation engine
│ ├── Board.py # Grid and tile management
│ ├── Tile.py # Individual tile implementation
│ ├── Tetromino.py # Shape representation
│ └── DataHandler.py # Data collection utilities
├── Exp_.py # Experimental scripts
├── Comparison_Extended/ # Comparison experiment results
├── Convergence_Analysis/ # Convergence study results
├── Fail_Analysis/ # Failure analysis results
├── Fault_Tolerance/ # Fault tolerance results
├── Optimization/ # Optimization results
└── Media/ # Visualizations and figures
```

## Research Applications

This framework is designed for research in:
- **Swarm Robotics**: Distributed control strategies
- **Self-Assembly**: Collective shape formation
- **Emergent Behavior**: How local interactions create global patterns
- **Robustness**: System performance under failures
- **Optimization**: Parameter tuning for complex systems

## Contributing

The project follows a modular design where new behaviors can be easily added to `Behaviors.py` and new experiments can be created following the pattern in `Exp_*.py` files.

## License

This is a research project. Please contact the authors for usage permissions.