import random
from Environment.Simulator import Simulator
from Environment.Tile import Tile
from Behaviors import Behaviors
import json
import _config
from tqdm import tqdm
import sys
import numpy as np
from TunableParameters import TunableParameters
TunableParameters.set_params()

if __name__ == '__main__':
    import _folders
    experiment_name = '_Fail_Analysis'
    _folders.set_experiment_folders(experiment_name)

    seed = 0
    random.seed(seed)
    np.random.seed(seed)


    TILE_SIZE = 20
    setup = {
        'N' : 10,
        'TILE_SIZE' : TILE_SIZE,
        'object': True,
        'symbol': 'T',
        'target_shape': True,
        'show_tetromines' : False,
        'show_tetromino_contour' : True,
        
        'resolution': 2,

        'n_random_targets' : 0,
        'shuffle_targets': False,
        
        'delay': False,
        'visualize': False,

        'save_data': True,
        'data_tiles': False,
        'data_objet_target': False,
        'file_name': 'defaultname',

        'dead_tiles': 0,
        'save_animation': False,
        'max_iterations': 1000,
    }

    BEHAVIORS = [   Behaviors.InfDiff, 
                    Behaviors.Discrete,
                    Behaviors.Logistic,
                    Behaviors.Gaussian,
                    Behaviors.Fourier,
                 ]
    
    BEHAVIORS_NAMES = ['InfDiff', 
                       'Discrete', 
                       'Logistic', 
                       'Gaussian', 
                       'Fourier']
    
    SYMBOLS = ["I", "O", "T", "J", "L", "S", "Z"]
    SYMMETRIES = {"I": 180, 
                  "O": 90, 
                  "T": 360, 
                  "J": 360, 
                  "L": 360, 
                  "S": 180, 
                  "Z": 180}
    
    RUNS = 500
    SYMBOLS = [random.choice(["I", "O", "T", "J", "L", "S", "Z"]) for _ in range(RUNS)]
    ANGLES_OBJECT = [random.random()*360 for _ in range(RUNS)]
    ANGLES_TARGET = [random.random()*360 for _ in range(RUNS)]
    POSITIONS = [(random.random()*setup['N']*setup['TILE_SIZE'], random.random()*setup['N']*setup['TILE_SIZE']) for _ in range(RUNS)]


    results = {}
    def get_convergence_step(arr):
        if len(arr) <= 1: return 0
        last_value = arr[-1]
        for index, value in enumerate(reversed(arr)):
            if value != last_value: return len(arr) - index
        return len(arr)

    for behavior, behaviors_name in tqdm(zip(BEHAVIORS, BEHAVIORS_NAMES)):
        Tile.execute_behavior = behavior
        
        shapes = []
        target_angles = []
        initial_angles = []
        relative_initial_angles = []
        error_angles_symmetry = []

        for run in tqdm(range(RUNS)):
            setup['symbol'] = random.choice(SYMBOLS)

            simulator = Simulator(setup)

            simulator.tetromino.rect.center = POSITIONS[run]            
            simulator.tetromino.rotate(ANGLES_OBJECT[run], allow_max_rotation=False)
            simulator.target.set_angle(ANGLES_TARGET[run])
            run_data = simulator.run_simulation()
            
            #PERFORMANCE CALCULATION

            shape = run_data['SHAPE']
            target_angle = run_data['TARGET_ANGLE']            
            initial_angle = run_data['object_angle'][0]
            final_angle = run_data['object_angle'][-1]
            
            initial_angle = (initial_angle % 360)% SYMMETRIES[shape]
            target_angle = (target_angle % 360)% SYMMETRIES[shape]
            final_angle = (final_angle % 360)% SYMMETRIES[shape]


            # Calculate the relative initial angle and ensure it's in the range [-180, 180]
            relative_initial_angle = (target_angle - initial_angle) % 360
            if relative_initial_angle > 180:
                relative_initial_angle -= 360  # Shift into the range [-180, 180]

            # Calculate the error considering symmetrical rotations
            error_angle_symmetry = (target_angle - final_angle) % 360
            if error_angle_symmetry > 180:
                error_angle_symmetry -= 360  # Shift into the range [-180, 180]            
            
            
            shapes.append(shape)
            initial_angles.append(initial_angle)
            error_angles_symmetry.append(error_angle_symmetry)
            target_angles.append(target_angle)
            relative_initial_angles.append(relative_initial_angle)

            results[behaviors_name] = { 'SHAPES': shapes,
                                        'TARGET_ANGS': target_angles,
                                        'INITIAL_ANGLE': initial_angles,
                                        'RELATIVE_INITIAL_ANGLE': relative_initial_angles,
                                        'ER_ANGS': error_angles_symmetry,
                                        }


            results_path = f'{_folders.RESULTS_PATH}/results.json'
            with open(results_path, 'w') as file:
                json.dump(results, file, indent=4)        
            
    
