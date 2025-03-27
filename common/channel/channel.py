import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# This program generates a double exponential channel and plots the eye diagram

def double_exponential_channel(t_h, tau):
    """
    Generates a double exponential channel with a given time vector and tau value.
    Parameters:
    t_h (numpy.ndarray): The time vector for the channel.
    tau (float): The time constant for the channel.
    Returns:
    numpy.ndarray: The double exponential channel.
    """
  
    #channel (exponential) (with step function)
    h = np.exp(-(t_h-1)/tau)*np.heaviside(t_h-1, 1)
    h = np.convolve(h, h, mode='same')

    #normalize h
    h = h/np.sum(h)

    return h


if __name__ == "__main__":

    tau = 0.02
    OSR = 64
    strt_h = -5
    stop_h = 15
    npre = 2

    t_h = np.linspace(strt_h, stop_h, (stop_h-strt_h)*OSR)
    h = double_exponential_channel(t_h, tau)
    h_pulse = np.convolve(h, np.ones((OSR,)))
     
    pk_pulse = np.argmax(h_pulse)
    h_pulse = h_pulse[pk_pulse-npre*OSR:pk_pulse+4*OSR]

    t = np.linspace(0, len(h_pulse)/OSR, len(h_pulse))
    t = t - npre



    data = [-1, 1, -1, -1, -1]

    # All fonts bold and big!

    d_individual =np.zeros((len(data), len(data)*OSR + len(h_pulse)))
    for i in range(len(data)):
        d_individual[i,i*OSR:i*OSR+len(h_pulse)] = data[i] * h_pulse
        
    t = np.linspace(0, d_individual.shape[1]/OSR, d_individual.shape[1])
    t = t - npre

    #Create a subplot of plots for the channel - use subplot, make it look nice
    fig = plt.figure(figsize=(8, 2*5))
    gs = fig.add_gridspec(len(data), 1, hspace=0)
    axs = gs.subplots(sharex='col', sharey=True)

    # add grid but make it dashed, bold and alpha = 0.5
    for i in range(len(data)):
        axs[i].plot(t, d_individual[i], label='Data {}'.format(i), color='C{}'.format(i), linewidth=2.5)
        axs[i].grid(color='black', linestyle='--', linewidth=1.5, alpha=0.25)
        axs[i].set_yticks(np.arange(-0.5, 0.5+0.5, 0.5))
        axs[i].set_yticklabels(np.arange(-0.5, 0.5+0.5, 0.5), fontsize=12, fontweight='bold', math_fontfamily='dejavuserif')
        for spine in axs[i].spines.values():
            spine.set_linewidth(2)

    #Set x-ticks to be at unit intervals
    axs[-1].set_xticks(np.arange(-2, len(data)+4, 1))
    axs[-1].set_xticklabels(np.arange(-2, len(data)+4, 1), fontsize=12, fontweight='bold', math_fontfamily='dejavuserif')


    # Add a title and labels

    plt.savefig('2_individual_pulses.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # Create a stacked plot of the data - no area under the curve
    fig = plt.figure(figsize=(8, 2))
    ax = fig.add_subplot(111)
    for i in range(len(data)):
        plt.plot(t, np.sum(d_individual[0:i+1,:], axis=0), linewidth=2.5, label='Data {}'.format(i), zorder=len(data)-i+1)

    # overlap with bold dashed d_sum
    plt.plot(t, np.sum(d_individual, axis=0), 'k--', linewidth=5, label='Sum of Data', zorder=len(data)+2)
    ax.set_xticks(np.arange(-2, len(data)+4, 1))
    ax.set_xticklabels(np.arange(-2, len(data)+4, 1), fontsize=12, fontweight='bold', math_fontfamily='dejavuserif')
    ax.set_yticks(np.arange(-0.5, 0.5+0.5, 0.5))
    ax.set_yticklabels(np.arange(-0.5, 0.5+0.5, 0.5), fontsize=12, fontweight='bold', math_fontfamily='dejavuserif')

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.grid(color='black', linestyle='--', linewidth=1.5, alpha=0.25)
    
    plt.savefig('2_sum_of_pulses.pdf', dpi=300, bbox_inches='tight')
    plt.close()

