import json
import os
#import upper level folder to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np

import _colors
import _folders
import gzip
import sys

experiment_name=f'_Comparison_Extended'
_folders.set_experiment_folders(experiment_name)
import _config

def _correct_behavior_name(behavior):
    if behavior     == 'InfDiff':             return 'InfDiff'
    elif behavior   == 'Discrete':            return 'Discrete'
    elif behavior   == 'Logistic':            return 'Logistic'
    elif behavior   == 'Gaussian':            return 'Gaussian'
    elif behavior   == 'Fourier':             return 'Fourier'
    else: raise ValueError(f'Behavior {behavior} not recognized')

def _plot_histogram(results, metric, ylabel, filename, pallette, n_bins):
    max_value = max(np.max(results[behavior][metric]) for behavior in results)
    min_value = min(np.min(results[behavior][metric]) for behavior in results)

    bins = np.linspace(min_value, max_value, n_bins)
    for i, behavior in enumerate(results):
        distance_error = results[behavior][metric]
        mean = np.mean(distance_error)
        std = np.std(distance_error)
        plt.hist(distance_error, bins=bins, alpha=0.6, label=f'{_correct_behavior_name(behavior)}', facecolor=pallette[i], weights=100*np.ones_like(distance_error) / len(distance_error))
        plt.axvline(mean, color=pallette[i],  linewidth=1, label=f'$\mu:{mean:.2f}$,  $\sigma:{std:.2f}$', alpha=0.6)

    plt.legend(ncol=2, fontsize=8)

    plt.xlabel(ylabel)
    plt.ylabel('Frequency [%]')    
    file_path = f'{_folders.VISUALIZATIONS_PATH}/{filename}.png'
    plt.savefig(file_path, bbox_inches='tight', dpi=600)
    plt.clf()

def _plot_histogram_lines(results, metric, ylabel, filename, pallette, n_bins):
    max_value = max(np.max(results[behavior][metric]) for behavior in results)
    min_value = min(np.min(results[behavior][metric]) for behavior in results)

    bins = np.linspace(min_value, max_value, n_bins)
    for i, behavior in enumerate(results):
        distance_error = results[behavior][metric]
        #get frequency of each bin
        hist, bin_edges = np.histogram(distance_error, bins=bins)    
        mean = np.mean(distance_error)
        std = np.std(distance_error)
        plt.plot(bin_edges[:-1], hist, color=pallette[i], label=f'{_correct_behavior_name(behavior)}', linewidth=1.2)
        #plt.hist(distance_error, bins=bins, alpha=0.6, label=f'{_correct_behavior_name(behavior)}', facecolor=pallette[i], weights=100*np.ones_like(distance_error) / len(distance_error))
        #plt.axvline(mean, color=pallette[i],  linewidth=1, label=f'$\mu:{mean:.2f}$,  $\sigma:{std:.2f}$', alpha=0.6)

    plt.legend(ncol=2, fontsize=8)

    plt.xlabel(ylabel)
    plt.ylabel('Frequency [%]')    
    file_path = f'{_folders.VISUALIZATIONS_PATH}/{filename}.png'
    plt.savefig(file_path, bbox_inches='tight', dpi=600)
    plt.clf()

def _plot_box(results, metric, ylabel, filename, pallette):
    data = []
    for i, behavior in enumerate(results):
        angle_error = results[behavior][metric]
        data.append(angle_error)        
        
    linewidth = 1.2
    # Create the box plot
    box = plt.boxplot(data,
                patch_artist=True,
                showmeans=True,
                meanline=True,
                showfliers=False,
                boxprops=dict(facecolor='white', color='black', linewidth=linewidth),
                whiskerprops=dict(color='black', linewidth=linewidth),
                capprops=dict(color='black', linewidth=linewidth),
                medianprops=dict(color='black', linewidth=linewidth),
                meanprops=dict(color = 'black', linewidth=linewidth),
                labels=[_correct_behavior_name(behavior) for behavior in results])
    
    for patch, color in zip(box['boxes'], pallette):
        patch.set_facecolor(color)
    
    plt.ylabel(ylabel)
    file_path = f'{_folders.VISUALIZATIONS_PATH}/{filename}.png'
    plt.savefig(file_path, bbox_inches='tight', dpi=600)
    plt.clf()

def resultant_error_histogram(results):
    _colors.FIG_SIZE = (4, 2)
    plt.subplots(figsize=_colors.FIG_SIZE)

    pallette = _colors.create_palette(len(results.keys()))
    
    print(results.keys())

    METRICS = ['ER_POS', 'ER_ANG', 'COVERAGE', 'OPT']
    YLABELS = ['Position Error [tiles]', 'Angle Error\nSymmetry Free [$^\circ$]', 'Coverage [%]', 'Operation Period\n[update steps]']
    FILENAMES = [m + '_HIST' for m in METRICS]
    N_BINS = [17, 20, 20, 10, 15]
    for metric, ylabel, filename, n_bins in zip(METRICS, YLABELS, FILENAMES, N_BINS):
        _plot_histogram(results, metric, ylabel, filename, pallette, n_bins)

def resultant_error_histogram_lines(results):
    _colors.FIG_SIZE = (4, 2)
    plt.subplots(figsize=_colors.FIG_SIZE)

    pallette = _colors.create_palette(len(results.keys()))
    
    print(results.keys())

    METRICS = ['ER_POS', 'ER_ANG', 'COVERAGE', 'OPT']
    YLABELS = ['Position Error [tiles]', 'Angle Error\nSymmetry Free [$^\circ$]', 'Coverage [%]', 'Operation Period\n[update steps]']
    FILENAMES = [m + '_LINE' for m in METRICS]
    N_BINS = [17, 20, 20, 10, 15]
    for metric, ylabel, filename, n_bins in zip(METRICS, YLABELS, FILENAMES, N_BINS):
        _plot_histogram_lines(results, metric, ylabel, filename, pallette, n_bins)

def resultant_error_box(results):
    _colors.FIG_SIZE = (4, 2)
    plt.subplots(figsize=_colors.FIG_SIZE)

    pallette = _colors.create_palette(len(results.keys()))
    
    print(results.keys())

    METRICS = ['ER_POS', 'ER_ANG', 'COVERAGE', 'OPT']
    YLABELS = ['Position Error [tiles]', 'Angle Error\nSymmetry Free [$^\circ$]', 'Coverage [%]', 'Operation Period\n[update steps]']
    FILENAMES = [m + '_BOX' for m in METRICS]

    for metric, ylabel, filename in zip(METRICS, YLABELS, FILENAMES):
        _plot_box(results, metric, ylabel, filename, pallette)



def write_table(results):

    METRICS = ['ER_POS', 'ER_ANGS' ,'COVERAGE', 'OPT']
    YLABELS = ['Position Error [tiles]', 'Angle Error\nSymmetry Free [$^\circ$]', 'Coverage [%]', 'Operation Period\n[update steps]']
    BEHAVIORS = ['InfDiff', 'Discrete', 'Logistic', 'Gaussian', 'Fourier']


    table = {}

    for behavior in results:
        table[behavior] = {}
        for metric in METRICS:
            mean = np.mean(results[behavior][metric])
            std = np.std(results[behavior][metric])
            table[behavior][metric] = {'mean': mean, 'std': std}
            print(f'{behavior} {metric} {mean:.2f} {std:.2f}')
    
    behaviors = BEHAVIORS
    metrics = METRICS
    metrics_names = YLABELS

    string = '\\begin{table}\n\centering\n\caption{Performance metric results.}\n\\label{tab1}\n\\begin{tabular}{|l|l|l|l|}\n\hline\n'
    string += 'Metric & Behavior & $\mu$ & $\sigma$ \\\\ \hline\n'
    
    for metric, metric_n in zip(metrics, metrics_names):
        means = []
        stds = []
        for behavior in behaviors:
            means.append(table[behavior][metric]['mean'])
            stds.append(table[behavior][metric]['std'])

        if metric_n == 'Coverage [%]':
            best_index = means.index(max(means))
            second_best_index = means.index(sorted(means)[-2])
        else:
            best_index = means.index(min(means))
            second_best_index = means.index(sorted(means)[1])

        size = '\small'
        # Behavior
        metric_name = metric_n.replace('[%]', '$[\%]$')
        line = f'{metric_name} & '
        line += '\\begin{tabular}[c]{@{}l@{}}'
        for i, behavior in enumerate(behaviors):
            behavior = _correct_behavior_name(behavior)
            if i == best_index:
                line += f'\\textbf{{{behavior}}}\\\\'
            elif i == second_best_index:
                line += f'\\textit{{{behavior}}}\\\\'
            else:
                line += f'{behavior}\\\\'
        
        line = line[:-2]
        line += '\\end{tabular} & '
        
        # Mean
        line += '\\begin{tabular}[c]{@{}l@{}}'
        for i, mean in enumerate(means):
            if i == best_index:
                line += f'\\textbf{{{mean:.2f}}}\\\\'
            elif i == second_best_index:
                line += f'\\textit{{{mean:.2f}}}\\\\'
            else:
                line += f'{mean:.2f}\\\\'
        line = line[:-2]
        line += '\\end{tabular} & '
        
        # Std
        line += '\\begin{tabular}[c]{@{}l@{}}'
        for i, std in enumerate(stds):
            if i == best_index:
                line += f'\\textbf{{{std:.2f}}}\\\\'
            elif i == second_best_index:
                line += f'\\textit{{{std:.2f}}}\\\\'
            else:
                line += f'{std:.2f}\\\\'
        line = line[:-2]
        line += '\\end{tabular} \\\\'
        line += '\hline\n'
        string += line

    string += '\\multicolumn{3}{l}{\\footnotesize{$^{*}$Best results in \\textbf{bold}, second-best results in \\textit{italics}.}} \\'
    string += '\\\end{tabular}\n\end{table}'

    #save to file
    table_path = f'{_folders.RESULTS_PATH}/table.txt'
    with open(table_path, 'w') as file:
        file.write(string)


if __name__ == '__main__':
    filename = f'{_folders.RESULTS_PATH}/results.json'
    with open(filename, 'r') as file:
        results = json.load(file)
    
    #resultant_error_box(results)
    #resultant_error_histogram(results)
    #resultant_error_histogram_lines(results)
    write_table(results)