import os
import pandas as pd
import numpy as np
import random
import math
def feature_normalize(data):
    mu = np.mean(data)
    std = np.std(data)
    return (data - mu)/std

def reshape_vector(signal, outputsize):
    return signal.reshape((outputsize[0],outputsize[1]))

# second order attenuation signal
def h(t, one_side=True):
    eL = 0.02
    eR = 0.005
    f1 = 2000
    if t < 0:
        if one_side:
            ht=0
        else:
            ht = math.exp((-eL/math.sqrt(1-eL**2))*(2*math.pi*f1*t)**2)*math.cos(2*math.pi*f1*t)
    else:
        ht = math.exp((-eR/math.sqrt(1-eR**2))*(2*math.pi*f1*t)**2)*math.cos(2*math.pi*f1*t)
    return ht

# Generate bearing signals
def bear_generate(signal_length, fs, bear_para, j):
    signal_clean = np.zeros(signal_length)
    deltat=1/fs
    time= np.linspace(0,(signal_length-1)/fs, signal_length)

    N = bear_para['N']                                    # Number of rolling element
    d = bear_para['d']                                    # Diameter of rolling element
    D = bear_para['D']                                    # Pitch diameter
    alpha = bear_para['alpha']                            # Contact angle
    fr = bear_para['fr']                                  # Rolling frequency
    fi = N/2 * fr * (1 + d/D * np.cos(alpha))             # Characteristic frequency of Inner ring failure 
    fo = N/2 * fr * (1 - d/D * np.cos(alpha))             # Characteristic frequency of Outer ring failure 
    fb = 1/2 * D/d * fr * (1 - (d/D * np.cos(alpha))**2)  # The Rolling frequency of the rolling element relative to the holding frame
    fc = 1/2 * fr * (1 - d/D * np.cos(alpha))             # The Rolling frequency of the cage relative to the outer ring

    FTF = fr/2*(1-d/D*np.cos(alpha))
    T_list = [1/fi, 1/fo, 1/fb, 1/fc]
    T = T_list[j]
    phase = [1/fr, T, 1/FTF][j]
    t0=0 # np.random.uniform(-phase,0)
    for i in range(signal_length):
        for k in range(50):
            deltaTk=0
            signal_clean[i] = signal_clean[i] + h((i+1)*deltat-k*T-deltaTk-t0, one_side=True)
    
    if bear_para['mode'][j]=='inner':
        signal_clean = (1.5+np.cos(2*np.pi*fr*(time-t0)))*signal_clean
    if bear_para['mode'][j]=='ball':
        signal_clean = (1.5+np.sin(2*np.pi*FTF*(time-t0)))*signal_clean

    noise = np.random.normal(loc=0, scale=bear_para['sigma'], size=signal_length)
    # signal_clean = (2*signal_clean)/(signal_clean.max()-signal_clean.min())
    signal_noisy = signal_clean + noise
    return signal_clean, signal_noisy

# Generate gear signal
def gear_generate(signal_length, fs, gear_para, j):
    signal_clean = np.zeros(signal_length)
    noise = np.zeros(signal_length)
    deltat=1/fs
    time= np.linspace(0,(signal_length-1)/fs, signal_length)

    fr = gear_para['fr']
    fm = gear_para['fr']*gear_para['teeth_num']
    T = 1/fr
    t0=0
    for i in range(signal_length):
        for k in range(50):
            deltaTk=0
            signal_clean[i] = signal_clean[i] + h((i+1)*deltat-k*T-deltaTk-t0, one_side=False)
    if gear_para['mode'][j]=='local':
        signal_clean = signal_clean + 0.5*np.sin(2*np.pi*fm*(time-t0))
    if gear_para['mode'][j]=='distributed':
        signal_clean = np.cos(2*np.pi*fm*(time-t0))*(1.5+np.cos(2*np.pi*fr*(time-t0)))
    noise = np.random.normal(loc=0, scale=gear_para['sigma'], size=signal_length)
    signal_noisy = signal_clean + noise
    return signal_clean, signal_noisy

def data_prepare(sample_number, test_size, signal_length, Fs, bear_para, gear_para):

    tN = int(sample_number*(1-test_size))
    # train data
    data_1d, gth_1d, label = [], [], []
    bear_len, gear_len = len(bear_para['mode']), len(gear_para['mode'])
    print('prepare data...')
    for i in range(sample_number):
        bear_or_gear = np.random.randint(0,bear_len+gear_len)
        print(i, '/',sample_number)
        if bear_or_gear<bear_len:
            signal_clean, signal_noisy = bear_generate(signal_length, Fs, bear_para, bear_or_gear)
        else:
            signal_clean, signal_noisy = gear_generate(signal_length, Fs, gear_para, bear_or_gear-bear_len)
        data_1d.append(signal_noisy)
        gth_1d.append(signal_clean)
        label.append(bear_or_gear)
    
    train_gth_1d, test_gth_1d = np.array(gth_1d)[:tN,np.newaxis], np.array(gth_1d)[tN:,np.newaxis]
    train_1d, train_label = np.array(data_1d)[:tN,np.newaxis], np.array(label)[:tN]
    test_1d, test_label = np.array(data_1d)[tN:,np.newaxis], np.array(label)[tN:]

    dataset = {'train_1d':feature_normalize(train_1d), 'train_label':train_label, 'test_1d':feature_normalize(test_1d),
     'test_label':test_label, 'train_gth_1d':feature_normalize(train_gth_1d), 'test_gth_1d':feature_normalize(test_gth_1d)}
    return dataset

# -1 to 1 Normalization
def norm(signal):
    return (2*signal)/(signal.max()-signal.min())