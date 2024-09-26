import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import random
from Environment.Simulator import Simulator
import random
from Environment.Tile import Tile
from Behaviors import Behaviors
import json
import sys
import numpy as np
from TunableParameters import TunableParameters
from tqdm import tqdm
import EvolutionarStrategies

# SIMULATION
def simulate_individual():
        setup_0['symbol'] = random.choice(["I", "O", "T", "J", "L", "S", "Z"])
        setup_0['resolution'] = 2
        simulator = Simulator(setup_0)
        results = simulator.run_simulation()
        return results

def reward_function_individual():
    run_rewards = []
    for _ in range(5):
        try:
            results = simulate_individual()
            coverage = results['coverage'][-1]
        except:
            coverage = 0

        run_rewards.append(coverage)
    return np.mean(np.array(run_rewards))


def pre_process(name:str, params:np.array):
    if name == 'Discrete':
        params = abs(params)
        return params
    
    elif name == 'Logistic':
        params[0] = abs(params[0])
        return params
    
    elif name == 'Gaussian':
        params[0] = abs(params[0])
        return params
        
    elif name == 'Fourier':
        params[0] = abs(params[0])
        return params
    
    else: raise ValueError('Invalid name')
    
if __name__ == "__main__":
    import _folders
    _folders.set_experiment_folders('_Optimization')
    
    setup_0 = {
        'N' : 10,
        'TILE_SIZE' : 20,
        'object': True,
        'symbol': 'T',
        'target_shape': True,
        'show_tetromines' : False,
        'show_tetromino_contour' : False,
        
        'resolution': 2,

        'n_random_targets' : 0,
        'shuffle_targets': False,
        
        'delay': False,
        'visualize': False,

        'save_data': True,
        'data_tiles': False,
        'data_objet_target': False,
        'file_name': False,

        'dead_tiles': 0,
        'save_animation': False,
        'max_iterations': 600,
    }
    
    BEHAVIORS = [Behaviors.Discrete, 
                 Behaviors.Logistic, 
                 Behaviors.Gaussian, 
                 Behaviors.Fourier]
    
    PARAMETERS = [TunableParameters.DISCRETE_PARAMS,
                  TunableParameters.LOGISTIC_PARAMS,
                  TunableParameters.GAUSSIAN_PARAMS,
                  TunableParameters.FOURIER_PARAMS]
    
    NAMES = ['Discrete', 'Logistic', 'Gaussian', 'Fourier']
    
    for B, P, N in zip(BEHAVIORS, PARAMETERS, NAMES):
        Tile.execute_behavior = B
        file_path = f'{_folders.RESULTS_PATH}/Evolution_{N}.json'

        results = {
            'BEST': [],
            'REWARDS': []
        }
        solver = EvolutionarStrategies.CMAES(num_params=len(P), popsize=3, weight_decay=0.01, sigma_init=0.5)
        for g in range(5):
            solutions = solver.ask()
            fitness_list = np.zeros(solver.popsize)

            for i in range(solver.popsize):
                solutions[i] = pre_process(N, solutions[i])
                if N == 'Discrete':
                    TunableParameters.DISCRETE_PARAMS = solutions[i]
                elif N == 'Logistic':
                    TunableParameters.LOGISTIC_PARAMS = solutions[i]
                elif N == 'Gaussian':
                    TunableParameters.GAUSSIAN_PARAMS = solutions[i]
                elif N == 'Fourier':
                    TunableParameters.FOURIER_PARAMS = solutions[i]
                else: raise ValueError('Invalid name')

                fitness_list[i] = reward_function_individual()

            solver.tell(fitness_list)
            result = solver.result()

            best_params, best_reward, curr_reward, sigma = result[0], result[1], result[2], result[3]
            print(f'G:{g}, BEST PARAMS: {best_params.tolist()}, BEST REWARD: {best_reward}')

            results['BEST'].append(best_params.tolist())
            results['REWARDS'].append(fitness_list.tolist())
            with open(file_path, 'w') as f:
                json.dump(results, f)