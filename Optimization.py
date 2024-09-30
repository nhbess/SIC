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
def simulate_individual(symbol, position, angle, setup):
        setup['symbol'] = symbol
        simulator = Simulator(setup)
        simulator.tetromino.rect.center = position
        simulator.tetromino.angle = angle            
        results = simulator.run_simulation()
        return results

def reward_function_individual(SYMBOLS,POSITIONS,ANGLES,setup):
    run_rewards = []
    for i in range(len(SYMBOLS)):
        symbol = SYMBOLS[i]
        position = POSITIONS[i]
        angle = ANGLES[i]

        try:
            results = simulate_individual(symbol, position, angle, setup)
            coverage = results['coverage'][-1]
        except:
            coverage = 0
        run_rewards.append(coverage)
    return np.mean(np.array(run_rewards))


if __name__ == "__main__":

    import _folders
    _folders.set_experiment_folders('_Optimization')
    random.seed(0)
    np.random.seed(0)
    setup = {
        'N' : 20,
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


    N_GENERATIONS = 32
    POPULATION_SIZE = 16
    RUNS = 50

    SYMBOLS = [random.choice(["I", "O", "T", "J", "L", "S", "Z"]) for _ in range(RUNS)]
    POSITIONS = [(random.random()*setup['N']*setup['TILE_SIZE'], random.random()*setup['N']*setup['TILE_SIZE']) for _ in range(RUNS)]
    ANGLES = [random.random()*360 for _ in range(RUNS)]
    
            
    for B, P, N in zip(BEHAVIORS, PARAMETERS, NAMES):
        Tile.execute_behavior = B
        file_path = f'{_folders.RESULTS_PATH}/Evolution_{N}.json'

        results = {
            'BEST': [],
            'REWARDS': []
        }

        solver = EvolutionarStrategies.CMAES(num_params=len(P), popsize=POPULATION_SIZE, weight_decay=0.01, sigma_init=0.5)
        for g in range(N_GENERATIONS):
            solutions = solver.ask()
            fitness_list = np.zeros(solver.popsize)

            for i in range(solver.popsize):
                if N == 'Discrete':
                    solutions[i] = np.abs(solutions[i])
                    TunableParameters.DISCRETE_PARAMS = solutions[i]
                
                elif N == 'Logistic':
                    solutions[i][0] = abs(solutions[i][0])
                    TunableParameters.LOGISTIC_PARAMS = solutions[i]
                
                elif N == 'Gaussian':
                    solutions[i][0] = abs(solutions[i][0])
                    TunableParameters.GAUSSIAN_PARAMS = solutions[i]
                
                elif N == 'Fourier':
                    solutions[i][0] = abs(solutions[i][0])
                    TunableParameters.FOURIER_PARAMS = solutions[i]
                
                else: raise ValueError('Invalid name')

                fitness_list[i] = reward_function_individual(SYMBOLS,POSITIONS,ANGLES,setup)

            solver.tell(fitness_list)
            result = solver.result()

            best_params, best_reward, curr_reward, sigma = result[0], result[1], result[2], result[3]
            print(f'G:{g}, BEST PARAMS: {best_params.tolist()}, BEST REWARD: {best_reward}')

            results['BEST'].append(best_params.tolist())
            results['REWARDS'].append(fitness_list.tolist())
            with open(file_path, 'w') as f:
                json.dump(results, f)

    best_params = {}
    for N in NAMES:
        file_path = f'{_folders.RESULTS_PATH}/Evolution_{N}.json'
        with open(file_path, 'r') as f:
            results = json.load(f)
            best_params[N] = results['BEST'][-1]

    with open(f'{_folders.RESULTS_PATH}/best_params.json', 'w') as f:
        json.dump(best_params, f)