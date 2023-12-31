
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt

# Define the path to the folder
data_folder = "F:/postgraduateProject/inHead/pythonProject/data/day/D5"
# Specify the path to the folder where pictures are saved
save_folder = "F:/postgraduateProject/inHead/pythonProject/output_graph/waveletImage/day5"

# Get a list of all txt files
txt_files = [f for f in os.listdir(data_folder) if f.endswith(".txt")]

# Iterate through each txt file
for idx, txt_file in enumerate(txt_files):
    txt_file_path = os.path.join(data_folder, txt_file)

    data = np.loadtxt(txt_file_path, delimiter=',')

    # Extraction of wave number and intensity data
    wavenumber = data[:, 0]
    intensity = data[:, 1]

    #Continuous Wavelet Transform
    wavelet = 'cmor'  # Select the wavelet basis function, which can be adjusted according to the actual situation
    scales = np.arange(1, 128)  # Scale range, can be adjusted according to the actual situation

    coefficients, frequencies = pywt.cwt(intensity, scales, wavelet)
    # b_values = frequencies[0:]
    # print(f"b_values = frequencies[0, :]:{b_values}")

    # calculate Amplitude Spectrum
    amplitude_spectrum = np.abs(coefficients)

    # 绘制结果
    plt.imshow(amplitude_spectrum, aspect='auto', extent=[min(frequencies), max(frequencies), min(scales), max(scales)],
               cmap='jet', interpolation='bilinear')
    plt.colorbar(label='Amplitude')
    plt.title('Continuous Wavelet Transform Amplitude Spectrum')
    plt.xlabel('Frequency')
    plt.ylabel('Wavelet Scale')


    # plt.show()

    # Specify the file path and file name to save the image
    save_path = os.path.join(save_folder, f"D5-{idx}.png")

    # save
    plt.savefig(save_path)

    # close
    plt.close()

print("The graphic is drawn and saved to the specified folder.")

