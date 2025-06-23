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

    # 2. DEFINE SIMULATION PARAMETERS
    # Use a fixed seed to ensure all behaviors face the exact same starting conditions
    seed = np.random.randint(0, 1000000)
    print(f"Seed: {seed}")

    TILE_SIZE = 20
    setup = {
        'N' : 20,
        'TILE_SIZE' : TILE_SIZE,
        'object': True,
        'symbol': 'T',  # Use a single shape for consistent comparison
        'target_shape': True,
        'show_tetromines' : False,
        'show_tetromino_contour' : True,
        
        'resolution': 2,

        'n_random_targets' : 0,
        'shuffle_targets': False,
        
        'delay': False,
        'visualize': False, # Set to True to watch simulations. Runs slower.

        'save_data': True,
        'new_data_scheme': True,
        'file_name': 'defaultname',

        'dead_tiles': 0,
        # Save an animation (GIF) of the simulation for each behavior
        'save_animation': True, 
        'max_iterations': 100, # Set a max iteration to prevent infinite loops
        'early_stop': False
    }

    # 3. DEFINE BEHAVIORS TO COMPARE
    BEHAVIORS_TO_TEST = {
        'Discrete': Behaviors.Discrete,
        #'Logistic': Behaviors.Logistic,
        #'Gaussian': Behaviors.Gaussian,
        #'Fourier': Behaviors.Fourier,
    }

    # 4. RUN EXPERIMENT
    print(f"Starting Convergence Analysis Experiment for shape: {setup['symbol']}")
    
    # Load optimal parameters if they exist
    TunableParameters.set_params()
    TunableParameters.print_params()

    full_results = {}

    for name, behavior in tqdm(BEHAVIORS_TO_TEST.items(), desc="Running Behaviors"):
        print(f"\nRunning behavior: {name}")
        
        # Reset seed for each behavior to ensure identical starting conditions
        random.seed(seed)
        np.random.seed(seed)
        
        Tile.execute_behavior = behavior
        
        # Set a unique path for each animation
        animation_path = f'{_folders.VISUALIZATIONS_PATH}/{name}_convergence'
        current_setup = setup.copy()
        current_setup['save_animation'] = animation_path

        simulator = Simulator(current_setup)
        
        # The run_simulation method returns a dictionary with the time-series data
        run_data = simulator.run_simulation(save_sys_data=True)
        
        # Store the collected data
        if run_data:
            full_results[name] = run_data
        else:
            print(f"Warning: No data returned for behavior {name}. Simulation might have timed out or failed.")
            full_results[name] = {}


        # Save results intermittently
        results_path = f'{_folders.RESULTS_PATH}/convergence_results.json'
        with open(results_path, 'w') as file:
            # A simple way to handle potential numpy types in data is to convert them to lists
            # This is a basic conversion, more complex objects may need a custom encoder
            def default_converter(o):
                if isinstance(o, np.integer):
                    return int(o)
                if isinstance(o, np.floating):
                    return float(o)
                if isinstance(o, np.ndarray):
                    return o.tolist()
                raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

            json.dump(full_results, file, indent=4, default=default_converter)

    print(f"\nExperiment finished. Results saved to {results_path}") 

def plot_convergence_snapshots(timesteps_to_plot):
    """
    Plots snapshots of the simulation at specified timesteps for each behavior.
    Creates a grid of subplots where rows are behaviors and columns are timesteps.
    """
    # 1. LOAD DATA
    results_path = f'{_folders.RESULTS_PATH}/convergence_results.json'
    try:
        with open(results_path, 'r') as file:
            full_results = json.load(file)
    except FileNotFoundError:
        print(f"Results file not found at {results_path}. Run the simulation first.")
        return

    behaviors = list(full_results.keys())
    num_behaviors = len(behaviors)
    num_snapshots = len(timesteps_to_plot)

    if num_behaviors == 0:
        print("No behaviors found in results file.")
        return

    # Determine global min/max for signal values for consistent colormap scaling
    all_signals = []
    for behavior_name in behaviors:
        behavior_data = full_results[behavior_name]
        if behavior_data.get('results', {}).get('MEMBRANE_TILES'):
            for timestep_data in behavior_data['results']['MEMBRANE_TILES']:
                if 'signal' in timestep_data and timestep_data['signal']:
                    all_signals.extend(timestep_data['signal'])

    min_signal, max_signal = (min(all_signals), max(all_signals)) if all_signals else (0, 1)

    # Create a normalizer and a colormap
    norm = plt.Normalize(vmin=min_signal, vmax=max_signal)
    cmap = plt.get_cmap('viridis')

    # Assuming all setups are the same, use the first one for board parameters
    first_behavior_name = behaviors[0]
    setup = full_results[first_behavior_name]['setup']
    tile_size = setup['TILE_SIZE']
    board_size_pixels = setup['BOARD_SIZE'] * tile_size

    # 2. CREATE PLOT
    fig, axes = plt.subplots(num_behaviors, num_snapshots, 
                             figsize=(num_snapshots * 3, num_behaviors * 3.5), 
                             squeeze=False, constrained_layout=True)
    fig.suptitle('Convergence Snapshots', fontsize=20, weight='bold')

    for i, behavior_name in enumerate(behaviors):
        behavior_data = full_results[behavior_name]
        
        axes[i, 0].set_ylabel(behavior_name, fontsize=14, rotation=90, labelpad=20, weight='bold')

        for j, timestep in enumerate(timesteps_to_plot):
            ax = axes[i, j]

            if i == 0:
                ax.set_title(f't = {timestep}', fontsize=14, weight='bold')

            ax.set_xlim(0, board_size_pixels)
            ax.set_ylim(0, board_size_pixels)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Make axis lines thicker
            for spine in ax.spines.values():
                spine.set_linewidth(2)

            # Draw grid
            for x in range(0, board_size_pixels + 1, tile_size):
                ax.axhline(y=x, color=COLORS.GRID, linestyle='-', linewidth=0.5)
                ax.axvline(x=x, color=COLORS.GRID, linestyle='-', linewidth=0.5)

            # Plot target shape
            target_tiles = behavior_data['setup']['TARGET_TILES']
            for x_coord, y_coord in zip(target_tiles['x'], target_tiles['y']):
                rect = patches.Rectangle((x_coord * tile_size, y_coord * tile_size), tile_size, tile_size,
                                         linewidth=1, edgecolor='none', facecolor=COLORS.TARGET, alpha=0.7)
                ax.add_patch(rect)

            # Reconstruct and plot target polygon
            target_center = behavior_data['setup']['TARGET_CENTER']
            target_angle = behavior_data['setup']['TARGET_ANGLE']
            shape = behavior_data['setup']['SHAPE']
            resolution = behavior_data['setup']['RESOLUTION']
            
            # Reconstruct the target tetromino
            temp_target = Tetromino(shape, tile_size, resolution=resolution)
            temp_target.set_angle(target_angle)
            temp_target.rect.center = (target_center[0], target_center[1])

            # Get and draw target polygon
            target_polygon = temp_target.mask.outline()
            if target_polygon:
                abs_target_polygon = [[p[0] + temp_target.rect.x, p[1] + temp_target.rect.y] for p in target_polygon]
                target_polygon_patch = patches.Polygon(abs_target_polygon, closed=True, edgecolor=f'#{_colors.PALETTE[0]}', facecolor='none', linewidth=2)
                ax.add_patch(target_polygon_patch)

            # Plot object shape at timestep
            if timestep < len(behavior_data['results']['MEMBRANE_TILES']):
                membrane_tiles = behavior_data['results']['MEMBRANE_TILES'][timestep]
                
                if 'signal' in membrane_tiles and membrane_tiles['signal']:
                    for k, (x_coord, y_coord) in enumerate(zip(membrane_tiles['x'], membrane_tiles['y'])):
                        signal = membrane_tiles['signal'][k]
                        color = cmap(norm(signal))
                        rect = patches.Rectangle((x_coord * tile_size, y_coord * tile_size), tile_size, tile_size,
                                                 linewidth=1, edgecolor='k', facecolor=color)
                        ax.add_patch(rect)
                else: # Fallback for old data
                    for x_coord, y_coord in zip(membrane_tiles['x'], membrane_tiles['y']):
                        rect = patches.Rectangle((x_coord * tile_size, y_coord * tile_size), tile_size, tile_size,
                                                 linewidth=1, edgecolor='k', facecolor=COLORS.OBJECT)
                        ax.add_patch(rect)
            else:
                ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

            # Reconstruct and plot object polygon
            ts_data = behavior_data['results']['time_series']
            if timestep < len(ts_data['object_center_x']):
                shape = behavior_data['setup']['SHAPE']
                resolution = behavior_data['setup']['RESOLUTION']
                center_x = ts_data['object_center_x'][timestep]
                center_y = ts_data['object_center_y'][timestep]
                angle = ts_data['object_angle'][timestep]

                # Reconstruct the tetromino
                temp_tetromino = Tetromino(shape, tile_size, resolution=resolution)
                temp_tetromino.set_angle(angle)
                temp_tetromino.rect.center = (center_x, center_y)

                # Get and draw polygon
                object_polygon = temp_tetromino.mask.outline()
                if object_polygon:
                    abs_polygon = [[p[0] + temp_tetromino.rect.x, p[1] + temp_tetromino.rect.y] for p in object_polygon]
                    polygon_patch = patches.Polygon(abs_polygon, closed=True, edgecolor=_colors.YELLOW, facecolor='none', linewidth=2)
                    ax.add_patch(polygon_patch)

    # Add a colorbar for the signal values
    if all_signals:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axes.ravel().tolist(), label='Signal', shrink=0.8, orientation='vertical')

    # Save the figure
    save_path = f'{_folders.VISUALIZATIONS_PATH}/convergence_snapshots.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConvergence snapshot plot saved to {save_path}")

if __name__ == '__main__':
    run_simulation()
    
    # Set the correct experiment folder before plotting
    
    _folders.set_experiment_folders('_Convergence_Analysis')
    
    plot_convergence_snapshots(timesteps_to_plot=[0, 50, 100])