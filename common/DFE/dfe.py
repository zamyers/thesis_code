import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from common.channel.channel import double_exponential_channel as h_func
from common.eye_diagram.eye_diagram import plot_eye_diagram
from common.FFE.ffe import calculate_ffe_coefficients, plot_ffe_coefficients

# This program generates an eye diagram for a given channel and data post DFE
# Written 70% by ChatGPT, 30% by me

def slicer(data, threshold=0, type='NRZ'):
    """
    Slices the data based on the given threshold.
    Parameters:
    data (numpy.ndarray): The data to be sliced.
    threshold (float): The threshold for slicing.
    Returns:
    numpy.ndarray: The sliced data.
    """
    if type == 'NRZ':
        return np.where(data > threshold, 1, -1)
    elif type == 'PAM4':
        return np.where(data > threshold, 1, -1)

if __name__ == "__main__":

    num_taps = 16
    # Parameters
    start_d = 0
    stop_d = 1000000
    start_h = -20
    stop_h = 40
    OSR = 32
    tau = 0.75
    sig = 0.05
    num_dfe_taps = 3
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
    data_o = data_o + np.random.normal(0, sig, size=data_o.shape)
    

    # calculate the ffe coefficients


    h_pulse = np.convolve(h, np.ones(OSR,))
    pk_pulse = np.argmax(h_pulse)
    print(pk_pulse)


    plt.plot(h_pulse[pk_pulse-3*OSR::OSR], label='Channel Pulse')
    plt.title('Channel Pulse')
    plt.show()

    dfe_coefficients = h_pulse[pk_pulse+OSR:pk_pulse+(num_dfe_taps+1)*OSR:OSR]/h_pulse[pk_pulse]
    

    ffe_coefficients = np.convolve(calculate_ffe_coefficients(h_pulse[pk_pulse-3*OSR::OSR], num_taps, sig=sig), h_pulse[pk_pulse:pk_pulse+(num_dfe_taps+1)*OSR:OSR]/h_pulse[pk_pulse])
    ffe_coefficients = ffe_coefficients[:num_taps]
    plt.plot(h_pulse[::OSR])
    plt.title('Channel Impulse Response')
    plt.show()
    #Zero Order Hold 
    ffe_coeff_osr = np.pad(np.reshape(ffe_coefficients, (num_taps, 1)), (0, OSR-1), 'constant', constant_values=0).flatten()

    plt.stem(h_pulse[pk_pulse-3*OSR::OSR], label='Channel Pulse')
    plt.title('Channel Pulse')
    plt.show()
    
    plt.stem(np.convolve(h_pulse[pk_pulse-3*OSR::OSR], ffe_coefficients), label='Convolution of Channel and FFE Coefficients')
    plt.title('Convolution of Channel and FFE Coefficients')
    plt.show()

    plt.stem(ffe_coefficients)
    plt.title('FFE Coefficients')
    plt.show()

    plt.stem(dfe_coefficients, label='DFE Coefficients')
    plt.title('DFE Coefficients')
    plt.show()
    # Create the DFE

    data_o = np.convolve(data_o, ffe_coeff_osr, mode='same')

    sampled_data = data_o[::OSR]
    dfe_thresh   = np.zeros(sampled_data.shape)
    sliced_data  = np.zeros(sampled_data.shape)

    for ii in range(num_dfe_taps+1, len(sampled_data)):
        dfe_thresh [ii] = np.dot(dfe_coefficients, sliced_data[ii-num_dfe_taps:ii][::-1])
        sliced_data[ii] = slicer(sampled_data[ii] - dfe_thresh[ii])


    # Recreate Oversampled Data

    dfe_thresh_osr = np.repeat(dfe_thresh, OSR)
    data_o_dfe = data_o[:-OSR//2] - dfe_thresh_osr[OSR//2:]

    #plt.plot(data_o_dfe)
    #plt.plot(dfe_thresh_osr[OSR//2:])
    #plt.plot(data_o)
    #plt.title('DFE Output')
    #plt.show()

    # Plot the eye diagram
    data_dfe = pd.DataFrame({'Time': t_d_o[:-OSR//2], 'Voltage': data_o_dfe})
    plot_eye_diagram(data_dfe, 'Voltage', UI=2, OSR=32, trgt=1e-4, pdf=True, pdf_title='2_dfe_1')
    plot_ffe_coefficients(ffe_coefficients, pdf=True, pdf_title='2_dfe_1_ffe_coeffs')