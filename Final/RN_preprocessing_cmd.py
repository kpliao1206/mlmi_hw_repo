import numpy as np
import matplotlib.pyplot as plt

def open_switch_test(file_path='', plot=False):
    data = np.loadtxt(file_path, skiprows=1)

    pulse_num = data[:, 0]
    resistance = data[:, 1]
    pulse_amp = data[:, 2]

    if plot:
        plt.figure()
        plt.plot(pulse_num, resistance, '-o')
        plt.ylim((1.44, 1.52))
        plt.title(file_path, fontsize=8)
        plt.xlabel('Pulse number')
        plt.ylabel('Resistance')
        plt.show()
    
    return pulse_num, resistance, pulse_amp


def scale_array(arr, min_val, max_val):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    scaled_arr = (arr - arr_min) / (arr_max - arr_min)  
    scaled_arr = scaled_arr * (max_val - min_val) + min_val 
    return scaled_arr


def min_scale(arr):
    arr_min = np.min(arr)
    return arr - arr_min


def mean_scale(arr):
    arr_mean = (np.max(arr)+np.min(arr)) / 2
    return arr - arr_mean


pn, r, _ = open_switch_test('Switch_Test_Ix=16E-3A_Hx=0E+0Oe_Pulse Width=80.000000E-6s_Read I=500E-6A _7.txt', plot=False)
r_period = r[50:200]
r_period_scale = scale_array(np.array(r_period), -2, 2)
np.save('Synapse_weight/weight2_3period', r_period_scale)