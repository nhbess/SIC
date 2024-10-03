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
    experiment_name = '_Fault_Tolerance'
    _folders.set_experiment_folders(experiment_name)
    
    TunableParameters.set_params()

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

    SYMMETRIES = {"I": 180, 
                "O": 90, 
                "T": 360, 
                "J": 360, 
                "L": 360, 
                "S": 180, 
                "Z": 180}


    BEHAVIORS = [   Behaviors.InfDiff, 
                    Behaviors.Discrete,
                    Behaviors.Logistic,
                    Behaviors.Gaussian,
                    Behaviors.Fourier,
                 ]
    
    NAMES = ['InfDiff', 
             'Discrete', 
             'Logistic', 
             'Gaussian', 
             'Fourier']
    
    FAULTY_TILES = [i/10 for i in range(0, 10)]
    SYMBOLS = ["I", "O", "T", "J", "L", "S", "Z"]

    runs_per_percent = 100

    results = {}
    for behavior, behaviors_name in tqdm(zip(BEHAVIORS, NAMES)):
        Tile.execute_behavior = behavior
        results[behaviors_name] = []

        behavior_results = {}
        for percent in tqdm(FAULTY_TILES):
            setup['dead_tiles'] = percent
            
            pos_error = []
            ang_error = []
            angs_error = []
            for run in tqdm(range(runs_per_percent)):
                setup['symbol'] = random.choice(SYMBOLS)
                simulator = Simulator(setup)
                run_data = simulator.run_simulation()
                
                #PERFORMANCE CALCULATION

                shape = run_data['SHAPE']
                target_center = np.array(run_data['TARGET_CENTER'])/TILE_SIZE
                target_angle = run_data['TARGET_ANGLE']
                final_center = np.array([run_data['object_center_x'][-1],run_data['object_center_y'][-1]])/TILE_SIZE
                final_angle = run_data['object_angle'][-1]
                error_position = np.linalg.norm(target_center - final_center)
                error_angle = abs(target_angle - final_angle)
                
                target_angle = target_angle % 360
                final_angle = final_angle % 360

                error_angle = abs(target_angle - final_angle)
                if error_angle > 180: error_angle = 360 - error_angle


                target_angle = run_data['TARGET_ANGLE'] % SYMMETRIES[shape]
                final_angle = run_data['object_angle'][-1] % SYMMETRIES[shape]

                error_angle_symmetry = abs(target_angle - final_angle)
                if error_angle_symmetry > SYMMETRIES[shape]/2: error_angle_symmetry = SYMMETRIES[shape] - error_angle_symmetry


                pos_error.append(error_position)
                ang_error.append(error_angle)
                angs_error.append(error_angle_symmetry)

            behavior_results[percent] = {'POS_ERROR': pos_error, 
                                         'ANG_ERROR': ang_error,
                                         'ANGS_ERROR': angs_error}
        results[behaviors_name] = behavior_results

    results_path = f'{_folders.RESULTS_PATH}/faulty_data.json'
    with open(results_path, 'w') as file:
        json.dump(results, file, indent=4)
    
    
