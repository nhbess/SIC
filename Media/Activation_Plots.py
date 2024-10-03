import os
import sys
#add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

import _colors
import _folders
from TunableParameters import TunableParameters

FIG_SIZE = (5*0.75, 4*0.5)

def format_number(x):
            if np.isclose(x % np.pi, 0):  # Check if the number is a multiple of pi
                factor = int(x // np.pi)
                if factor == 0:
                    return "0"
                elif factor == 1:
                    return r"$\pi$"
                else:
                    return fr"${factor}\pi$"
            elif np.isclose(x % (np.pi/2), 0):  # Check if the number is a multiple of pi/2
                factor = int(x // (np.pi/2))
                if factor == 1:
                    return r"$\frac{\pi}{2}$"
                else:
                    return fr"${factor}\frac{{\pi}}{{2}}$"
            elif np.isclose(x % (np.pi/4), 0):  # Check if the number is a multiple of pi/4
                factor = int(x // (np.pi/4))
                if factor == 1:
                    return r"$\frac{\pi}{4}$"
                else:
                    return fr"${factor}\frac{{\pi}}{{4}}$"
            return f'{x:.2f}' if x % 1 else f'{int(x)}'  # Default formatting for other numbers

def plot_logistic():
    fig = plt.figure(figsize=FIG_SIZE)

    params = [[0.5, np.pi,    10, 0.8],
            [0.9, np.pi*0.8,      50, 0.5],
            [0.33,   np.pi/2,    1000, 0.89],
            TunableParameters.LOGISTIC_PARAMS,
            ]
        
    palette = _colors.create_palette(len(params))
    S = np.linspace(0, 1, 1000)

    for i, param in enumerate(params):
        LAMBDA  = param[0]
        SPAN    = param[1]
        SLOPE   = param[2]
        SHIFT   = param[3]

        A = SPAN/(1+np.exp(-SLOPE*(S - SHIFT)))
        
        
        #$\lambda$={format_number(param[0])}
        if i == len(params)-1:
            linestyle='--'
        else:
            linestyle='-'
        label = f'$P$={format_number(param[1])} $L$={format_number(param[2])} $H$={format_number(param[3])}'
        plt.plot(S, A, label=label, color=palette[i], linestyle=linestyle)

    plt.ylabel('$\\alpha$ [rad]')
    plt.xlabel('$s$')
    plt.legend(fontsize=8, framealpha=0.7, loc='upper left')
    #title
    plt.title('Logistic')
    #save
    plt.savefig(f'{_folders.MEDIA_PATH}/Logistic.png', dpi=300, bbox_inches='tight')
    plt.close()
    #plt.show()

def plot_gaussian():
    fig = plt.figure(figsize=FIG_SIZE)

    params = [[0.01, 0.2,      0.1, 1],
              [0.01, 0.1,      0.05, 0.5],
              [0.01, 0.3,      0.5, 0.8],
              TunableParameters.GAUSSIAN_PARAMS,
            ]
    
    
    palette = _colors.create_palette(len(params))
    S = np.linspace(0, 1, 1000)

    for i, param in enumerate(params):
        LAMBDA = param[0]
        A = param[1]
        B = param[2]
        C = param[3]
        
        ANGLE = np.exp(-np.power((S-C),2)/(np.power(A,2))) * (S-C)/(np.power(A,2))*B    
        
        #$\lambda$={format_number(param[0])} 
        label = f'$A$={format_number(param[1])} $B$={format_number(param[2])} $C$={format_number(param[3])}'
        if i == len(params)-1:
            linestyle='--'
        else:
            linestyle='-'

        plt.plot(S, ANGLE, label=label, color=palette[i], linestyle=linestyle)

    plt.ylabel('$\\alpha$ [rad]')
    plt.xlabel('$s$')
    plt.legend(fontsize=8, framealpha=0.7)#, loc='upper right')
    #title
    plt.title('Gaussian')
    plt.savefig(f'{_folders.MEDIA_PATH}/Gaussian.png', dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()

def plot_fourier():
    fig = plt.figure(figsize=FIG_SIZE)
    S = np.linspace(0, 2*np.pi, 1000)
    np.random.seed(456)

    params = [TunableParameters.FOURIER_PARAMS + np.random.uniform(-1, 1, 1 + 1 + 2*TunableParameters.FOURIER_TERMS)*0.5 for _ in range(3)]
    params.append(TunableParameters.FOURIER_PARAMS)
    palette = _colors.create_palette(len(params))
    
    for i, param in enumerate(params):
        def compute_alpha(omega, s):
                    a0 = param[1]
                    a_n = param[2:2 + TunableParameters.FOURIER_TERMS]
                    b_n = param[2 + TunableParameters.FOURIER_TERMS:]
    
                    alpha = a0  # Start with the zeroth term
                    # Change the loop to use range(1, TunableParameters.TERMS + 1) instead of len(Fourier_PARAMS) + 1
                    for n in range(1, TunableParameters.FOURIER_TERMS + 1):
                        alpha += a_n[n - 1] * np.cos(n * omega) + b_n[n - 1] * np.sin(n * omega)
                    alpha *= s  # Scale by s
                    return alpha
        
          
        ANGLE = compute_alpha(S, 1)
        ANGLE = np.clip(ANGLE, -np.pi, np.pi)
        if i == len(params)-1:
            linestyle='--'
        else:
            linestyle='-'
        plt.plot(S, ANGLE, color=palette[i], linestyle=linestyle)


    plt.ylabel('$\\alpha$ [rad]')
    plt.xlabel('$\omega$ [rad]')


    plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    #write the y ticks in radians
    plt.yticks([-np.pi*0.75, -np.pi/2, -np.pi/8], 
               [r'$-\frac{3\pi}{4}$', r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{8}$'])
    plt.title('Fourier')
    plt.savefig(f'{_folders.MEDIA_PATH}/Fourier.png', dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()


def plot_lambdas():
    NAMES = ['Discrete', 'Logistic', 'Gaussian', 'Fourier']
    LAMBDAS = [TunableParameters.DISCRETE_PARAMS[0], 
               TunableParameters.LOGISTIC_PARAMS[0], 
               TunableParameters.GAUSSIAN_PARAMS[0], 
               TunableParameters.FOURIER_PARAMS[0]]
    
    N_UPDATE_STEPS = 50
    UPDATE_STEPS = np.arange(N_UPDATE_STEPS)
    pallete = _colors.create_palette(len(LAMBDAS))
    plt.figure(figsize=(6, 2))
    plt.axvline(x=N_UPDATE_STEPS//2, color='black', linestyle='--')

    for i, l in enumerate(LAMBDAS):
        S = [0]
        for t in UPDATE_STEPS:
            I = int(t < N_UPDATE_STEPS//2)
            s = S[-1]*l + (1-l)*I
            S.append(s)

        S = S[0:-1]
        plt.plot(UPDATE_STEPS,S, label=f"{NAMES[i]} $\lambda$ = {np.round(l,2)}", color=pallete[i])


    plt.text(N_UPDATE_STEPS*0.25, 0.2, '$I = 1$', fontsize=12)
    plt.text(N_UPDATE_STEPS*0.75, 0.2, '$I = 0$', fontsize=12)
    plt.legend()
    plt.xlabel('Update Step')
    plt.ylabel('$s$')
    plt.savefig(f'{_folders.MEDIA_PATH}/Lambdas.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    from TunableParameters import TunableParameters
    TunableParameters.set_params()

    plot_lambdas()
    plot_logistic()
    plot_gaussian()
    plot_fourier()