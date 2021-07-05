# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
from urllib import request

#Datasets para previsão



def get_dataframe(filename, url, sep=";", compression='infer'):
    """
    This method check if filename already exists, read the file and return its data.
    If the file don't already exists, it will be downloaded and decompressed.

    :param filename: dataset local filename
    :param url: dataset internet URL
    :param sep: CSV field separator
    :param compression: type of compression
    :return:  Pandas dataset
    """

    request.urlretrieve(url, filename)
    return pd.read_csv(filename, sep=sep, compression=compression)

    """

    tmp_file = Path(filename)

    if tmp_file.is_file():
        return pd.read_csv(filename, sep=sep, compression=compression)
    else:
        request.urlretrieve(url, filename)
        return pd.read_csv(filename, sep=sep, compression=compression)
    """




def get_TAIEX():
    """
    Get the complete multivariate time series data.

    :return: Pandas DataFrame
    """
    dat = get_dataframe('TAIEX.csv.zip',
                               'https://github.com/arthurcaio92/pyT2FTS/raw/main/data/TAIEX.zip',
                               sep=",", compression='zip')
    dat["Date"] = pd.to_datetime(dat["Date"])
    return dat


def get_NASDAQ():
    """
    Get the complete multivariate time series data.

    :return: Pandas DataFrame
    """
    dat = get_dataframe('NASDAQ.csv.zip',
                               'https://github.com/arthurcaio92/pyT2FTS/raw/main/data/NASDAQ.zip',
                               sep=",", compression='zip')
    dat["Date"] = pd.to_datetime(dat["Date"])
    return dat


def get_SP500():
    """
    Get the complete multivariate time series data.

    :return: Pandas DataFrame
    """
    dat = get_dataframe('SP500.csv.zip',
                               'https://github.com/arthurcaio92/pyT2FTS/raw/main/data/SP500.zip',
                               sep=",", compression='zip')
    dat["Date"] = pd.to_datetime(dat["Date"])
    return dat


def get_Brent_Oil():
    """
    Get the complete multivariate time series data.

    :return: Pandas DataFrame
    """
    dat = get_dataframe('Brent-Oil.csv.zip',
                               'https://github.com/arthurcaio92/pyT2FTS/raw/main/data/Brent-Oil.zip',
                               sep=",", compression='zip')
    #dat["Date"] = pd.to_datetime(dat["Date"])
    
    dat_4000 = dat[4554:]  #Slices from 2001 for 4000 instances 
    dat_5000 = dat[3554:]  #Slices from 2001 for 5000 instances 
    
    return dat_4000





def mackey_glass_biblioteca(tau, n=10, beta=2, gamma=1, step=1000):
    x = [np.random.random() for i in range(tau)]
    for i in range(step):
        x.append(x[-1] + beta * x[-tau] / (1 + x[-tau] ** n) - gamma * x[-1])
    return x[tau:]




def mackey_glass_pyfts(b=.1, c=.2, tau=17, initial_values = 0, iterations=1000):
    '''
    
    Mackey, M. C. and Glass, L. (1977). Oscillation and chaos in physiological control systems.
    Science, 197(4300):287-289.
    dy/dt = -by(t)+ cy(t - tau) / 1+y(t-tau)^10


    Return a list with the Mackey-Glass chaotic time series.

    :param b: Equation coefficient
    :param c: Equation coefficient
    :param tau: Lag parameter, default: 17
    :param initial_values: numpy array with the initial values of y. Default: np.linspace(0.5,1.5,18)
    :param iterations: number of iterations. Default: 1000
    :return:
    '''
    
    initial_values = np.linspace(0.5,1.5, tau+1)
    y = initial_values.tolist()

    for n in np.arange(len(y)-1, iterations+100):
        y.append(y[n] - b * y[n] + c * y[n - tau] / (1 + y[n - tau] ** 10))

    return y[100:]





def mackey_glass_nolitsa(length=1000, x0=None, a=0.2, b=0.1, c=10.0, tau=30.0,
                 n=1000, sample=0.46, discard=250):
    """Generate time series using the Mackey-Glass equation.
    Generates time series using the discrete approximation of the
    Mackey-Glass delay differential equation described by Grassberger &
    Procaccia (1983).
    Parameters
    ----------
    length : int, optional (default = 10000)
        Length of the time series to be generated.
    x0 : array, optional (default = random)
        Initial condition for the discrete map.  Should be of length n.
    a : float, optional (default = 0.2)
        Constant a in the Mackey-Glass equation.
    b : float, optional (default = 0.1)
        Constant b in the Mackey-Glass equation.
    c : float, optional (default = 10.0)
        Constant c in the Mackey-Glass equation.
    tau : float, optional (default = 23.0)
        Time delay in the Mackey-Glass equation.
    n : int, optional (default = 1000)
        The number of discrete steps into which the interval between
        t and t + tau should be divided.  This results in a time
        step of tau/n and an n + 1 dimensional map.
    sample : float, optional (default = 0.46)
        Sampling step of the time series.  It is useful to pick
        something between tau/100 and tau/10, with tau/sample being
        a factor of n.  This will make sure that there are only whole
        number indices.
    discard : int, optional (default = 250)
        Number of n-steps to discard in order to eliminate transients.
        A total of n*discard steps will be discarded.
    Returns
    -------
    x : array
        Array containing the time series.
    """
    sample = int(n * sample / tau)
    grids = n * discard + sample * length
    x = np.empty(grids)

    if not x0:
        x[:n] = 0.5 + 0.05 * (-1 + 2 * np.random.random(n))
    else:
        x[:n] = x0

    A = (2 * n - b * tau) / (2 * n + b * tau)
    B = a * tau / (2 * n + b * tau)

    for i in range(n - 1, grids - 1):
        x[i + 1] = A * x[i] + B * (x[i - n] / (1 + x[i - n] ** c) +
                                   x[i - n + 1] / (1 + x[i - n + 1] ** c))
    return x[n * discard::sample]



from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal

def awgn(s,SNRdB,L=1):
    """
    
    METODO PARA ADICIONAR RUIDO EM DECIBEIS
    
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
"""
    gamma = 10**(SNRdB/10) #SNR to linear scale
    if s.ndim==1:# if s is single dimensional vector
        P=L*sum(abs(s)**2)/len(s) #Actual power in the vector
    else: # multi-dimensional signals like MFSK
        P=L*sum(sum(abs(s)**2))/len(s) # if s is a matrix [MxN]
    N0=P/gamma # Find the noise spectral density
    if isrealobj(s):# check if input is real/complex object type
        n = sqrt(N0/2)*standard_normal(s.shape) # computed noise
    else:
        n = sqrt(N0/2)*(standard_normal(s.shape)+1j*standard_normal(s.shape))
    r = s + n # received signal
    return r 