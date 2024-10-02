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
    plt.legend(fontsize=9, framealpha=0.7, loc='upper left')
    #title
    plt.title('Logistic')
    #save
    plt.savefig(f'{_folders.MEDIA_PATH}/Logistic.png', dpi=300, bbox_inches='tight')
    plt.close()
    #plt.show()

def plot_bidirect():
    fig = plt.figure(figsize=FIG_SIZE)

    params = [[0.01, 0.2,      0.1, 1],
              [0.01, 0.1,      0.05, 0.5],
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
    plt.legend(fontsize=9, framealpha=0.7)#, loc='upper right')
    #title
    plt.title('Gaussian')
    plt.savefig(f'{_folders.MEDIA_PATH}/Gaussian.png', dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()

def plot_fourier():
    fig = plt.figure(figsize=FIG_SIZE)
    S = np.linspace(0, 2*np.pi, 1000)
    
    def compute_alpha(omega, s):
                a0 = TunableParameters.FOURIER_PARAMS[1]
                a_n = TunableParameters.FOURIER_PARAMS[2:2 + TunableParameters.FOURIER_TERMS]
                b_n = TunableParameters.FOURIER_PARAMS[2 + TunableParameters.FOURIER_TERMS:]

                alpha = a0  # Start with the zeroth term
                # Change the loop to use range(1, TunableParameters.TERMS + 1) instead of len(Fourier_PARAMS) + 1
                for n in range(1, TunableParameters.FOURIER_TERMS + 1):
                    alpha += a_n[n - 1] * np.cos(n * omega) + b_n[n - 1] * np.sin(n * omega)
                alpha *= s  # Scale by s
                return alpha
    
    palette = _colors.create_palette(1)
      
    ANGLE = compute_alpha(S, 1)
    ANGLE = np.clip(ANGLE, -np.pi, np.pi)
    plt.plot(S, ANGLE, color=palette[0], linestyle='-', label='Fourier')


    plt.ylabel('$\\alpha$ [rad]')
    plt.xlabel('$\omega$ [rad]')


    plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    yticks = [-np.pi/2, -np.pi/3]
    yticks_labels = [r'$\frac{-\pi}{2}$', r'$\frac{-\pi}{3}$']    
    plt.yticks(yticks, yticks_labels)

    plt.title('Fourier')
    plt.savefig(f'{_folders.MEDIA_PATH}/Fourier.png', dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()


if __name__ == "__main__":
    from TunableParameters import TunableParameters
    TunableParameters.set_params()

    plot_logistic()
    plot_bidirect()
    plot_fourier()