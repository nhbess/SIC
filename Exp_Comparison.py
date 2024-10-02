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
if __name__ == '__main__':
    import _folders
    experiment_name = '_Comparison_Extended'
    _folders.set_experiment_folders(experiment_name)


    seed = 50
    random.seed(seed)
    np.random.seed(seed)


    TILE_SIZE = 20
    setup = {
        'N' : 20,
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

    results = {}
    def get_convergence_step(arr):
        if len(arr) <= 1: return 0
        last_value = arr[-1]
        for index, value in enumerate(reversed(arr)):
            if value != last_value: return len(arr) - index
        return len(arr)

    print("Starting experiment")
    TunableParameters.set_params()
    TunableParameters.print_params()

    for behavior, behaviors_name in tqdm(zip(BEHAVIORS, BEHAVIORS_NAMES)):
        Tile.execute_behavior = behavior
        
        shapes = []
        error_positions = []
        coverages = []
        error_angles = []
        error_angles_symmetry = []
        operation_times = []

        for run in tqdm(range(RUNS)):
            setup['symbol'] = random.choice(SYMBOLS)
            simulator = Simulator(setup)
            run_data = simulator.run_simulation()
            
            #PERFORMANCE CALCULATION
            shape = run_data['SHAPE']
            target_center = np.array(run_data['TARGET_CENTER'])/TILE_SIZE
            target_angle = run_data['TARGET_ANGLE']
            
            final_center = np.array([run_data['object_center_x'][-1],run_data['object_center_y'][-1]])/TILE_SIZE
            final_angle = run_data['object_angle'][-1]
            final_coverage = run_data['coverage'][-1]
            operation_time = get_convergence_step(run_data['coverage'])
            
            
            error_position = np.linalg.norm(target_center - final_center)
                        
            target_angle = target_angle % 360
            final_angle = final_angle % 360

            error_angle = abs(target_angle - final_angle)
            if error_angle > 180: error_angle = 360 - error_angle


            target_angle = run_data['TARGET_ANGLE'] % SYMMETRIES[shape]
            final_angle = run_data['object_angle'][-1] % SYMMETRIES[shape]

            error_angle_symmetry = abs(target_angle - final_angle)
            if error_angle_symmetry > SYMMETRIES[shape]/2: error_angle_symmetry = SYMMETRIES[shape] - error_angle_symmetry

            #print(f"Error: {error_angle}, Error with symmetry: {error_angle_symmetry}")

            shapes.append(shape)
            error_positions.append(error_position)
            coverages.append(final_coverage)
            error_angles.append(error_angle)
            error_angles_symmetry.append(error_angle_symmetry)
            operation_times.append(operation_time)

            results[behaviors_name] = { 'SHAPES': shapes,
                                        'ER_POS': error_positions,
                                        'ER_ANG': error_angles,
                                        'ER_ANGS': error_angles_symmetry,
                                        'COVERAGE': coverages,
                                        'OPT': operation_times}


            results_path = f'{_folders.RESULTS_PATH}/results.json'
            with open(results_path, 'w') as file:
                json.dump(results, file, indent=4)        
            
    
