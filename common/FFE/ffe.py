import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.linalg as la

from common.channel.channel import double_exponential_channel as h_func
from common.eye_diagram.eye_diagram import plot_eye_diagram

# This program generates an eye diagram for a given channel and data post FFE
# Written 70% by ChatGPT, 30% by me

def calculate_ffe_coefficients(channel, num_taps, sig=0.1):
    """
    Calculate the FFE coefficients based on the given channel.
    Parameters:
    channel (numpy.ndarray): The channel impulse response.
    num_taps (int): The number of taps for the FFE.
    Returns:
    numpy.ndarray: The FFE coefficients.
    """
    
    H = la.convolution_matrix(channel, num_taps)
    H_t =np.dot(H.T,la.pinv(np.dot(H,H.T) + sig**2*np.eye(H.shape[0])))

    ffe_coefficients = H_t[:, num_taps//2-1]
    return ffe_coefficients

def plot_ffe_coefficients(ffe_coefficients, pdf = False, pdf_title="default"):
    """
    Plot the FFE coefficients.
    Parameters:
    ffe_coefficients (numpy.ndarray): The FFE coefficients.
    """
    plt.stem(ffe_coefficients)
    plt.title('FFE Coefficients')
    if pdf:
        plt.savefig(pdf_title+'.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    # Parameters
    start_d = 0
    stop_d = 1000000
    start_h = -20
    stop_h = 40

    OSR = 32 
    tau = 0.75
    sig = 0.05
    num_taps = 16

    # Time vectors
    t_d = np.linspace(start_d, stop_d, (stop_d-start_d)*OSR)
    t_h = np.linspace(start_h, stop_h, (stop_h-start_h)*OSR)

    
    # Create the channel
    h = h_func(t_h, tau)

    # Create random data then upsample and first order hold
    data = np.random.randint(0, 2, size=(stop_d-start_d))*2 - 1
    data = np.repeat(data, OSR)

    # Convolve the data with the channel
    data_o = np.convolve(data, h, mode='same')

    # Trim the tail to remove zero values
    t_d_o  = t_d[0:len(data_o)-OSR*4]
    data_o = data_o[0:len(data_o)-OSR*4]
    
    # Add noise to the data
    data_o += np.random.normal(0, sig, size=data_o.shape)

    
    # Calculate FFE coefficients
    # Create Pulse Response
    h_pulse = np.convolve(h, np.ones((OSR,)))

    pk_loc = np.argmax(h_pulse)

    ffe_coefficients = calculate_ffe_coefficients(h_pulse[pk_loc-3*OSR::OSR], num_taps, sig=sig)
    plt.plot(h_pulse[::OSR])
    plt.title('Channel Impulse Response')
    plt.show()


    #Zero Order Hold 
    ffe_coeff_osr = np.pad(np.reshape(ffe_coefficients, (num_taps, 1)), (0, OSR-1), 'constant', constant_values=0).flatten()

    plt.plot(ffe_coeff_osr)
    plt.title('FFE Coefficients')
    plt.show()

    plt.plot(np.convolve(h_pulse, ffe_coeff_osr))
    plt.plot(h_pulse)
    plt.title('Convolution of Channel and FFE Coefficients')
    plt.show()

    # Apply FFE to the received signal
    ffe_output = np.convolve(data_o, ffe_coeff_osr, mode='same')

    
    # Plotting the eye diagram for the received signal after FFE
    df_ffe_output = pd.DataFrame({'Time': t_d_o[:len(ffe_output)], 'Voltage': ffe_output})
    
    plot_eye_diagram(df_ffe_output, 'Voltage', UI=2, OSR=32, trgt=1e-4, pdf=True, pdf_title='2_ffe_1')
    # Plot the FFE coefficients
    plot_ffe_coefficients(ffe_coefficients, pdf=True, pdf_title='2_ffe_1_coeffs')