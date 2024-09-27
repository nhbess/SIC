import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

import _colors
import _folders
import gzip
import sys

experiment_name=f'_Fail_Analysis'
_folders.set_experiment_folders(experiment_name)
import _config

SYMBOLS = ["L", "O", "T", "I", "S", "Z", "J"]

def _correct_behavior_name(behavior):
    if behavior     == 'InfDiff':             return 'InfDiff'
    elif behavior   == 'Discrete':            return 'Discrete'
    elif behavior   == 'Logistic':            return 'Logistic'
    elif behavior   == 'Gaussian':            return 'Gaussian'
    elif behavior   == 'Fourier':             return 'Fourier'
    else: raise ValueError(f'Behavior {behavior} not recognized')


def plot_fail_analysis(results):
    _colors.FIG_SIZE = (4, 4)  # Adjusting for polar plot size
    plt.subplots(figsize=_colors.FIG_SIZE, subplot_kw={'projection': 'polar'})  # Set projection to polar

    pallette = _colors.create_palette(len(results.keys()))    
    
    BEHAVIORS = list(results.keys())
    BEHAVIORS = BEHAVIORS[1:2]
    MARKERS = ['o', 's', 'D', 'v', '^', '<', '>', 'p', 'P', '*', 'h', 'H', 'X', 'D', 'd', '|', '_']
    print(BEHAVIORS)
    
    for i, behavior in enumerate(BEHAVIORS):
        shapes = results[behavior]['SHAPES']
        r_initial_angles = np.radians(results[behavior]['RELATIVE_INITIAL_ANGLE'])  # Convert degrees to radians
        er_angles = results[behavior]['ER_ANGS']
        
        plt.scatter(r_initial_angles, er_angles, alpha=0.5, label=f'{_correct_behavior_name(behavior)}', color=pallette[i], zorder = -i)
    
    plt.legend(ncol=2, fontsize=8, loc='upper left', bbox_to_anchor=(-0.2, 1.1))
    plt.ylabel('Angle Error Symmetry Free [°]', labelpad=25)

    plt.xlabel('Relative Initial Orientation [°]')    
    plt.gca().set_theta_zero_location('N')  # Set 0 degrees at the top (North)
    plt.gca().set_theta_direction(-1)  # Clockwise direction for angles

    plt.gca().set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    plt.gca().set_xticklabels(['0°', '45°', '90°', '135°', '180°', '-135°', '-90°', '-45°'])


    #plt.show()

    file_path = f'{_folders.VISUALIZATIONS_PATH}/RadialErrorBehavior.png'
    plt.savefig(file_path, bbox_inches='tight', dpi=600)
    #plt.clf()

def plot_fail_analysis_shapes(results):
    _colors.FIG_SIZE = (4, 4)  # Adjusting for polar plot size
    plt.subplots(figsize=_colors.FIG_SIZE, subplot_kw={'projection': 'polar'})  # Set projection to polar

    
    BEHAVIORS = list(results.keys())
    BEHAVIORS = BEHAVIORS[1:2]
    pallette = _colors.create_palette(len(SYMBOLS))    
    
    MARKERS = ['o', 's', 'D', 'v', '^', '<', '>', 'p', 'P', '*', 'h', 'H', 'X', 'D', 'd', '|', '_']
    
    for i, behavior in enumerate(BEHAVIORS):
        shapes = results[behavior]['SHAPES']
        r_initial_angles = np.radians(results[behavior]['RELATIVE_INITIAL_ANGLE'])  # Convert degrees to radians
        er_angles = results[behavior]['ER_ANGS']
        colors = [pallette[SYMBOLS.index(shape)] for shape in shapes]
        plt.scatter(r_initial_angles, er_angles, alpha=0.9, color=colors, zorder = -i)
    
    for symbol in SYMBOLS:
        plt.scatter([], [], label=symbol, color=pallette[SYMBOLS.index(symbol)], marker='o')

    plt.legend(ncol=2, fontsize=8, loc='upper left', bbox_to_anchor=(-0.2, 1.1))
    plt.ylabel('Angle Error [°]', labelpad=25)

    plt.xlabel('Relative Initial Orientation [°]')    
    plt.gca().set_theta_zero_location('N')  # Set 0 degrees at the top (North)
    plt.gca().set_theta_direction(-1)  # Clockwise direction for angles

    plt.gca().set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    plt.gca().set_xticklabels(['0°', '45°', '90°', '135°', '180°', '-135°', '-90°', '-45°'])


    #plt.show()

    file_path = f'{_folders.VISUALIZATIONS_PATH}/RadialErrorShape.png'
    plt.savefig(file_path, bbox_inches='tight', dpi=600)
    #plt.clf()


def plot_fail_analysis_boxplot(results):
    _colors.FIG_SIZE = (7, 4)  # Adjust size for the boxplot
    plt.subplots(figsize=_colors.FIG_SIZE)  # Set up a regular plot, not polar
    
    BEHAVIORS = list(results.keys())
    pallette = _colors.create_palette(len(SYMBOLS))    

    data = []       # Collect data for boxplots
    colors = []     # Store corresponding colors
    positions = []  # Store boxplot positions
    offset = 0      # Offset for grouping boxplots

    box_width = 0.6     # Width of each boxplot
    group_spacing = -2   # Spacing between behavior groups

    for i, behavior in enumerate(BEHAVIORS):
        shapes = results[behavior]['SHAPES']
        er_angles = results[behavior]['ER_ANGS']

        # Group data by shape
        shape_data = {}
        for shape, angle in zip(shapes, er_angles):
            if shape not in shape_data:
                shape_data[shape] = []
            shape_data[shape].append(angle)

        for j, symbol in enumerate(SYMBOLS):
            if symbol in shape_data:
                data.append(shape_data[symbol])
                colors.append(pallette[j])
                positions.append(offset + j * box_width)  # Set position for each boxplot in the group

        offset += len(SYMBOLS) + group_spacing  # Update offset for next behavior group

    # Create the boxplots
    linewidth = 1
    bp = plt.boxplot(data, 
                     positions=positions, 
                     widths=box_width, 
                     patch_artist=True, 
                     showmeans=True,
                     meanline=True,
                     showfliers=False,
                     whiskerprops=dict(color='black', linewidth=linewidth),
                     capprops=dict(color='black', linewidth=linewidth),
                     medianprops=dict(color='black', linewidth=linewidth),
                     meanprops=dict(color = 'black', linewidth=linewidth),
                     
                     
                     )

    # Set colors for each box
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Add x-ticks for behaviors
    mid_positions = [offset + (len(SYMBOLS) * box_width) / 2 for offset in range(0, int(len(BEHAVIORS) * (len(SYMBOLS) + group_spacing)), int(len(SYMBOLS) + group_spacing))]
    
    plt.xticks(mid_positions, [_correct_behavior_name(b) for b in BEHAVIORS], rotation=0, ha='right')

    # Add legend for the shapes
    for symbol in SYMBOLS:
        plt.scatter([], [], label=symbol, color=pallette[SYMBOLS.index(symbol)], marker='o')

    plt.legend( fontsize=8, loc='upper left', ncols=len(SYMBOLS))
    plt.ylabel('Angle Error Symmetry Free [°]', labelpad=3)
    plt.xlabel('Behavior')

    # Save the plot as an image
    file_path = f'{_folders.VISUALIZATIONS_PATH}/BoxplotErrorShape.png'
    plt.savefig(file_path, bbox_inches='tight', dpi=600)
    #plt.clf()


if __name__ == '__main__':
    filename = f'{_folders.RESULTS_PATH}/results.json'
    with open(filename, 'r') as file:
        results = json.load(file)
    
    plot_fail_analysis(results)
    plot_fail_analysis_shapes(results)
    plot_fail_analysis_boxplot(results)
    
    #write_table(results)