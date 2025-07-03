import json
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.patches as patches
import pygame

import _colors
import _folders
from Behaviors import Behaviors
from Environment.Simulator import Simulator
from Environment.Tile import Tile
from Environment.Tetromino import Tetromino
from TunableParameters import TunableParameters

# Global seed for reproducibility
GLOBAL_SEED = 1

SYMMETRIES = {
    'T': 360, 'L': 360, 'J': 360, 'S': 180, 'Z': 180, 'I': 180, 'O': 90
}

class COLORS:
    palette = _colors.create_palette(9, normalize=True)
    OBJECT = palette[4]
    TARGET = palette[0]
    GRID = '#d3d3d2'

def run_simulation():
    # 1. SETUP EXPERIMENT
    experiment_name = '_Convergence_Analysis'
    _folders.set_experiment_folders(experiment_name)

    # 2. SET GLOBAL SEED FOR REPRODUCIBILITY
    print(f"Global Seed: {GLOBAL_SEED}")
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    TILE_SIZE = 20
    setup = {
        'N': 20,
        'TILE_SIZE': TILE_SIZE,
        'object': True,
        'symbol': 'T',  # Fixed shape for all behaviors
        'target_shape': True,
        'show_tetromines': False,
        'show_tetromino_contour': True,
        
        'resolution': 2,

        'n_random_targets': 0,
        'shuffle_targets': False,
        
        'delay': False,
        'visualize': False,

        'save_data': True,
        'data_tiles': False,
        'data_objet_target': False,
        'file_name': 'defaultname',

        'dead_tiles': 0,
        'save_animation': False,
        'max_iterations': 250,
        'early_stop': False,  # Disable early stopping for convergence analysis
    }

    # 3. DEFINE BEHAVIORS TO COMPARE (same as Exp_Comparison)
    BEHAVIORS = [
        Behaviors.InfDiff, 
        Behaviors.Discrete,
        Behaviors.Logistic,
        Behaviors.Gaussian,
        Behaviors.Fourier,
    ]
    
    BEHAVIORS_NAMES = [
        'InfDiff', 
        'Discrete', 
        'Logistic', 
        'Gaussian', 
        'Fourier'
    ]
    
    SYMMETRIES = {
        "I": 180, 
        "O": 90, 
        "T": 360, 
        "J": 360, 
        "L": 360, 
        "S": 180, 
        "Z": 180
    }

    def calculate_metrics_over_time(run_data):
        """Calculate error metrics at each timestep"""
        shape = run_data['SHAPE']
        target_center = np.array(run_data['TARGET_CENTER']) / TILE_SIZE
        target_angle = run_data['TARGET_ANGLE']
        
        # Time series data
        centers = np.array([run_data['object_center_x'], run_data['object_center_y']]).T / TILE_SIZE
        angles = np.array(run_data['object_angle'])
        coverages = np.array(run_data['coverage'])
        
        # Calculate errors over time
        error_positions = np.linalg.norm(centers - target_center, axis=1)
        
        # Angle errors considering symmetry (this is the one that makes sense)
        target_angle_sym = target_angle % SYMMETRIES[shape]
        angles_sym = angles % SYMMETRIES[shape]
        error_angles_symmetry = np.abs(target_angle_sym - angles_sym)
        error_angles_symmetry = np.minimum(error_angles_symmetry, SYMMETRIES[shape] - error_angles_symmetry)
        
        return {
            'error_positions': error_positions,
            'error_angles_symmetry': error_angles_symmetry,
            'coverages': coverages,
            'timesteps': list(range(len(coverages)))
        }

    # 4. RUN EXPERIMENT
    print("Starting Convergence Analysis Experiment")
    print("Single run per behavior with fixed initial conditions")
    
    TunableParameters.set_params()
    TunableParameters.print_params()

    full_results = {}

    for behavior, behavior_name in tqdm(zip(BEHAVIORS, BEHAVIORS_NAMES), desc="Running Behaviors"):
        print(f"\nRunning behavior: {behavior_name}")
        
        # Reset to same seed for each behavior to ensure identical starting conditions
        random.seed(GLOBAL_SEED)
        np.random.seed(GLOBAL_SEED)
        
        Tile.execute_behavior = behavior
        
        # Use fixed shape T for all behaviors
        simulator = Simulator(setup)
        run_data = simulator.run_simulation()
        
        if run_data and 'coverage' in run_data:
            metrics = calculate_metrics_over_time(run_data)
            
            # Store single run data
            full_results[behavior_name] = {
                'error_positions': metrics['error_positions'],
                'error_angles_symmetry': metrics['error_angles_symmetry'],
                'coverages': metrics['coverages'],
                'timesteps': metrics['timesteps']
            }
        else:
            print(f"Warning: No data returned for behavior {behavior_name}")
            full_results[behavior_name] = {}

    # 5. SAVE RESULTS
    results_path = f'{_folders.RESULTS_PATH}/convergence_results.json'
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for behavior_name, data in full_results.items():
        if data:  # Check if data exists
            json_results[behavior_name] = {
                'error_positions': data['error_positions'].tolist(),
                'error_angles_symmetry': data['error_angles_symmetry'].tolist(),
                'coverages': data['coverages'].tolist(),
                'timesteps': data['timesteps']
            }
        else:
            json_results[behavior_name] = {}
    
    with open(results_path, 'w') as file:
        json.dump(json_results, file, indent=4)

    print(f"\nExperiment finished. Results saved to {results_path}")
    
    return full_results

def plot_convergence_metrics():
    """Plot metrics over time for all behaviors"""
    # Load data
    results_path = f'{_folders.RESULTS_PATH}/convergence_results.json'
    try:
        with open(results_path, 'r') as file:
            full_results = json.load(file)
    except FileNotFoundError:
        print(f"Results file not found at {results_path}. Run the simulation first.")
        return

    behaviors = list(full_results.keys())
    # Removed 'error_angles' and kept only the symmetry-aware one
    metrics = ['error_positions', 'error_angles_symmetry', 'coverages']
    metric_labels = ['Position Error [Tiles]', 'Angle Error Symmetry-Free [°]', 'Coverage [%]']
    
    # Create subplots (3 metrics instead of 4)
    X = 3
    fig, axes = plt.subplots(1, X, figsize=(4*X, X))
    
    # Use the proper color palette implementation
    colors = _colors.create_palette(len(behaviors), normalize=True)
    
    # Define unique linestyles for each behavior
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  # solid, dashed, dashdot, dotted, custom
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        
        for j, behavior_name in enumerate(behaviors):
            behavior_data = full_results[behavior_name]
            
            if behavior_data and metric in behavior_data:
                # Single run data - no averaging needed
                metric_values = np.array(behavior_data[metric])
                timesteps = np.arange(len(metric_values))
                
                # Plot single curve for each behavior
                linestyle = linestyles[j % len(linestyles)]
                ax.plot(timesteps, metric_values, label=behavior_name, color=colors[j], 
                       linewidth=2, linestyle=linestyle)
            else:
                print(f"Warning: No data for {behavior_name} - {metric}")
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel(label)
        #ax.set_title(f'{label}')
        ax.legend()
    
    plt.tight_layout()
    
    # Save the plot
    save_path = f'{_folders.VISUALIZATIONS_PATH}/convergence_plot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Convergence plot saved to {save_path}")

if __name__ == '__main__':
    # Run simulation
    run_simulation()
    
    # Plot results
    _folders.set_experiment_folders('_Convergence_Analysis')
    plot_convergence_metrics()