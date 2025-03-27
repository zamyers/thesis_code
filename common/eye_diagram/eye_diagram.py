import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from common.channel.channel import double_exponential_channel as h_func

def plot_eye_diagram(data, y_label, title=None, UI=2, OSR=32, trgt=1e-4, pdf=False, pdf_title="default"):

    #reshape time to be mod UI
    data['Time'] = data['Time'] % UI
    #shift the time to be centered around 0
    data['Time'] = data['Time'] - UI/2
    #Use Histogram to plot the eye diagram, with a black background
    sns.set_style("dark")

    sns.histplot(data, x='Time', y='Voltage', bins=128, pmax=1, cmap='viridis',alpha=0.9, pthresh=trgt)
    # Set the title and labels - all text bold
    if title:
        plt.title(title, fontsize=16, fontweight='bold')
    # Label the UI
    plt.xlabel('UI', fontsize=14, fontweight='bold')
    plt.ylabel(y_label , fontsize=14, fontweight='bold')

    # Grid - add dark bold grid lines on top of the plot
    plt.grid(color='black', linestyle='--', linewidth=2, alpha=1)

    #trim plot boundaries to be exactly +/- UI/2
    plt.xlim(-UI/2, UI/2)
    # Show the plot


    # Save the plot
    if pdf:
        plt.savefig(pdf_title+'.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":

    # 0 : tau = 0.25
    # 1 : tau = 0.45
    # 5 : tau = 0.75

    tau_sel = [0.25, 0.45, 0.5, 0.575, 0.65, 0.75]
    idx = 0
    start_d = 0
    stop_d = 1000000
    start_h = -20
    stop_h = 40
    sig = 0.05
    OSR = 32 
    tau = tau_sel[idx]
    #time 
    t_d = np.linspace(start_d, stop_d, (stop_d-start_d)*OSR)
    t_h = np.linspace(start_h, stop_h, (stop_h-start_h)*OSR)

    
    #Create the channel
    h = h_func(t_h, tau)

    pk_idx = np.argmax(np.convolve(h, np.ones(OSR,)))

    #Create random data then upsample and first order hold
    data = np.random.randint(0, 2, size=(stop_d-start_d))*2 - 1
    data = np.repeat(data, OSR)

    #Convolve the data with the channel
    data_o = np.convolve(data, h, mode='same')

    #Trim the tail to remove zero values
    t_d_o  = t_d[0:len(data_o)-OSR*16]
    data_o = data_o[0:len(data_o)-OSR*16]
    data_o = data_o + np.random.normal(0, sig, size=data_o.shape)


    #Plot 16 UIs of Data - with each UI a different disctint color with vertical lines at every UI
    # Make all the font bold 
    plt.figure(figsize=(24, 3))
    for i in range(8):
        plt.plot(t_d_o[i*OSR:(i+1)*OSR], data_o[pk_idx+i*OSR- OSR//2:pk_idx+(i+1)*OSR- OSR//2], color=sns.color_palette()[i%8], linestyle='-', linewidth=3)
        plt.axvline(x=t_d_o[i*OSR], color='black', linestyle='--', linewidth=2)

    plt.axvline(x=t_d_o[8*OSR], color='black', linestyle='--', linewidth=2)

    # Set the title and labels - all text bold

    plt.xlim(0, 8)

    # Save the plot as a pdf 
    plt.savefig('2_waveform.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(3, 3))
    #Plot the same previous 16 UI but overlapping
    for i in range(8):
        plt.plot(t_d_o[0:OSR], data_o[pk_idx+i*OSR- OSR//2:pk_idx+(i+1)*OSR- OSR//2], color=sns.color_palette()[i%8], linestyle='-', linewidth=3)
    
    plt.savefig('2_waveform_overlap.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    #Create a dataframe for the eye diagram
    df = pd.DataFrame({'Time': t_d_o, 'Voltage': data_o})

    plot_eye_diagram(df, 'Voltage (V)', UI=2, OSR=32, trgt=1e-4, pdf=True, pdf_title=f'2_raw_{idx}')

    h_pulse = np.convolve(h, np.ones(OSR,))

    pk_pulse = np.argmax(h_pulse)
    h_pulse = h_pulse[pk_pulse-3*OSR:pk_pulse+6*OSR]

    t = np.linspace(0, len(h_pulse)/OSR, len(h_pulse))
    t = t - 3

    # Normal - not dark 
    sns.set_style("white")
    plt.figure(figsize=(6, 3))

    # Plot with xticks every 1 UI 
    plt.plot(t, h_pulse, label='Channel Pulse', linewidth=3)
    # Add O's every 1 UI
    plt.plot(np.arange(-3, 6, 1), h_pulse[::OSR], 'o', label='Channel Pulse', color='black', markersize=4)
    # Add verticle lines that connect the O's to the x axis
    for i in range(9):
        plt.plot([i-3, i-3], [0, h_pulse[i*OSR]], 'k', linewidth=2, alpha=0.75)
    #Add grid lines
    plt.grid(color='black', linestyle='--', linewidth=1.5, alpha=0.25)
    plt.xlim(-3, 6)
    plt.ylim(-0.1, 1.1)
    plt.xticks(np.arange(-3, 6, 1), fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.savefig(f'2_channel_{idx}.pdf', dpi=300, bbox_inches='tight')
    plt.close()
