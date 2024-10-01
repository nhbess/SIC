import numpy as np
import json

class TunableParameters:
    DISCRETE_PARAMS = np.zeros(2) # Lambda + tau
    LOGISTIC_PARAMS = np.zeros(4) # Lambda + Span + Slope + Shift
    GAUSSIAN_PARAMS = np.zeros(4) # Lambda + A + B + C
    
    FOURIER_TERMS = 3
    FOURIER_PARAMS = np.zeros(1 + 1 + 2*FOURIER_TERMS)  # Lambda + a0 + (ai, bi) for i in TERMS
    

    @staticmethod
    def set_params():
        best_params_path = '_Optimization/__Results/best_params.json'
        with open(best_params_path, 'r') as file:
            best_params = json.load(file)

        TunableParameters.DISCRETE_PARAMS = np.array(best_params['Discrete'])
        TunableParameters.LOGISTIC_PARAMS = np.array(best_params['Logistic'])
        TunableParameters.GAUSSIAN_PARAMS = np.array(best_params['Gaussian'])
        TunableParameters.FOURIER_PARAMS = np.array(best_params['Fourier'])

    @staticmethod
    def print_params():
        print('Discrete:', TunableParameters.DISCRETE_PARAMS)
        print('Logistic:', TunableParameters.LOGISTIC_PARAMS)
        print('Gaussian:', TunableParameters.GAUSSIAN_PARAMS)
        print('Fourier:', TunableParameters.FOURIER_PARAMS)

TunableParameters.print_params()