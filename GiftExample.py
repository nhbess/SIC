import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import sys

environment_folder = 'Environment'
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), environment_folder))
import random
import numpy as np  
from Environment.Simulator import Simulator
from Environment.Tile import Tile
from Behaviors import Behaviors
from TunableParameters import TunableParameters

def run_simulation(behavior, name, seed):
    random.seed(seed)
    np.random.seed(seed)

    setup_0 = {
        'N' : 20,
        'TILE_SIZE' : 20,
        'object': True,
        'symbol': 'T',
        'target_shape': True,
        'show_tetromines' : False,
        'show_tetromino_contour' : True,
        
        'resolution': 5,

        'n_random_targets' : 0,
        'shuffle_targets': False,
        
        'delay': False,
        'visualize': True,

        'save_data': False,
        'data_tiles': True,
        'data_objet_target': False,
        'file_name': 'test',

        'dead_tiles': 0,
        'save_animation': name,
        'max_iterations': 500,
    }
    
    
    symbol = random.choice(["I", "O", "T", "J", "L", "S", "Z"])
    symbol = 'T'
    setup_0['symbol'] = symbol
    setup_0['resolution'] = 2
    simulator = Simulator(setup_0)
    simulator.run_simulation()

if __name__ == "__main__":
    seed = random.randint(0, 1000000)    
    print(f'Random seed: {seed}')

    TunableParameters.set_params()

    behaviors = [Behaviors.Gaussian, Behaviors.Logistic, Behaviors.Fourier, Behaviors.Discrete, Behaviors.InfDiff] 
    names = ['Gaussian', 'Logistic', 'Fourier', 'Discrete', 'InfDiff']

    for behavior, name in zip(behaviors, names):
        Tile.execute_behavior = behavior
        run_simulation(behavior, name, seed = seed)


