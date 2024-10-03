import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
experiment_name=f'_Fault_Tolerance'

import _colors
import _folders
_folders.set_experiment_folders(experiment_name)

def _correct_behavior_name(behavior):
    if behavior     == 'InfDiff':             return 'InfDiff'
    elif behavior   == 'Discrete':            return 'Discrete'
    elif behavior   == 'Logistic':            return 'Logistic'
    elif behavior   == 'Gaussian':            return 'Gaussian'
    elif behavior   == 'Fourier':            return 'Fourier'
    else: raise ValueError(f'Behavior {behavior} not recognized')

def resultant_error():
    filename = f'{_folders.RESULTS_PATH}/faulty_data.json'
    with open(filename, 'r') as file:
        results = json.load(file)
    print(results.keys())
    
    results_by_dead_tile = {}

    BEHAVIORS = list(results.keys())
    #BEHAVIORS = BEHAVIORS[0:2]

    for behavior in BEHAVIORS:
        err_pos = []
        err_ang = []
        err_angs = []
        
        for percent in results[behavior]:
            print(percent)
            err_pos.append(results[behavior][percent]['POS_ERROR'])
            err_angs.append(results[behavior][percent]['ANGS_ERROR'])

        
        err_pos = np.array(err_pos)
        err_ang = np.array(err_ang)
        err_angs = np.array(err_angs)

        results_by_dead_tile[behavior] = {
            'POS_ERROR': err_pos,
            'ANGS_ERROR': err_angs,
        }

    
    def _plot_error(results, metric, ylabel, filename):
        pallette = _colors.create_palette(len(results_by_dead_tile))  
        plt.subplots(figsize=(4, 2))
        for i, behavior in enumerate(results):
            pos_error = results[behavior][metric]
            pos_error_means = np.mean(pos_error, axis=1)
            pos_error_std = np.std(pos_error, axis=1)
            X = np.arange(len(pos_error_means))
            X = X * 10

            plt.scatter(X, pos_error_means, label=_correct_behavior_name(behavior), color=pallette[i])
            plt.plot(X, pos_error_means, color=pallette[i])

            # Calculate upper and lower bounds for the standard deviation
            upper_bound = pos_error_means + pos_error_std
            lower_bound = pos_error_means - pos_error_std

            # Plot the upper and lower bounds as dashed lines
            #plt.plot(X, upper_bound, linestyle='--', color=pallette[i], alpha=0.5)
            #plt.plot(X, lower_bound, linestyle='--', color=pallette[i], alpha=0.5)    
        
        plt.legend()
        plt.xlabel('Dead Tiles [%]')
        plt.ylabel(ylabel)

        file_path = f'{_folders.VISUALIZATIONS_PATH}/{filename}.png'
        plt.savefig(file_path, bbox_inches='tight', dpi=600)
        plt.clf()

    _plot_error(results_by_dead_tile, 'POS_ERROR', 'Position Error [tiles]', 'ROBUST_POS')
    #_plot_error(results_by_dead_tile, 'ANG_ERROR', 'Angle Error [$^\circ$]', 'ROBUST_ANG')
    _plot_error(results_by_dead_tile, 'ANGS_ERROR', 'Symmetry Free \nAngle Error [$^\circ$]', 'ROBUST_ANGS')

    

if __name__ == '__main__':
    resultant_error()