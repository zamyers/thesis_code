import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.linalg as la

from common.channel.channel import double_exponential_channel as h_func
from common.eye_diagram.eye_diagram import plot_eye_diagram
from common.FFE.ffe import calculate_ffe_coefficients


if __name__ == "__main__":
    # Parameters
    start_d = 0
    stop_d = 1000
    start_h = -20
    stop_h = 40

    OSR = 64
    tau = 0.125
    sig = 0.05
    num_taps = 16

    # Time vectors
    t_d = np.linspace(start_d, stop_d, (stop_d-start_d)*OSR)
    t_h = np.linspace(start_h, stop_h, (stop_h-start_h)*OSR)

    # Generate channel
    channel = h_func(t_h, tau)

    # Calculate FFE coefficients
    # Create Pulse Response
    h_pulse = np.convolve(channel, np.ones((OSR,)))

    pk_loc = np.argmax(h_pulse)

    ffe_coefficients = calculate_ffe_coefficients(h_pulse[pk_loc-3*OSR::OSR], num_taps, sig=sig)
  

    pam4_data = (np.random.randint(0, 4, size=(stop_d-start_d))*2 - 3)/3
    pam4_data = np.repeat(pam4_data, OSR)
    pam4_data_o = np.convolve(pam4_data, channel, mode='same')
    t_d_o = t_d[0:len(pam4_data_o)-OSR*4]
    pam4_data_o = pam4_data_o[0:len(pam4_data_o)-OSR*4]
    pam4_data_o = pam4_data_o + np.random.normal(0, sig, size=pam4_data_o.shape)
   


  
    # Plot eye diagram
    data = pd.DataFrame({'Time': t_d_o, 'Voltage': pam4_data_o})

    plot_eye_diagram(data, y_label='Voltage', title=None, UI=2, OSR=OSR, trgt=1e-3, pdf=True, pdf_title="2_pam4_raw_0")


    ffe_coeff_osr = np.pad(np.reshape(ffe_coefficients, (num_taps, 1)), (0, OSR-1), 'constant', constant_values=0).flatten()

    ffe_output = np.convolve(pam4_data_o, ffe_coeff_osr, mode='same')
    t_d_ffe = t_d_o[OSR*2:len(ffe_output)-OSR*4]
    ffe_output = ffe_output[OSR*2:len(ffe_output)-OSR*4]



    df_ffe_output = pd.DataFrame({'Time': t_d_ffe, 'Voltage': ffe_output})
    plot_eye_diagram(df_ffe_output, 'Voltage', UI=2, OSR=OSR, trgt=1e-3, pdf=True, pdf_title='2_pam4_ffe_0')


    