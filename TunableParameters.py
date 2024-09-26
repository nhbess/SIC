import numpy as np

class TunableParameters:
    #Discrete
    DISCRETE_PARAMS = np.random.rand(2) # Lambda + tau

    # Logistic
    LOGISTIC_PARAMS = np.random.rand(4) # Lambda + Span + Slope + Shift

    # Gaussian
    GAUSSIAN_PARAMS = np.random.rand(4) # Lambda + A + B + C

    # Fourier
    TERMS = 9
    FOURIER_PARAMS = np.random.rand(2 + 2 * TERMS) # Lambda + a0 + (ai, bi) for i in TERMS
