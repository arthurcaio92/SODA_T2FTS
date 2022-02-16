# -*- coding: utf-8 -*-
from SODA_T2FTS.SODA_T2FTS.sliding_window import run_sliding_window
from SODA_T2FTS.SODA_T2FTS.datasets import get_TAIEX,get_NASDAQ,get_Brent_Oil,get_SP500
import numpy as np
import pandas as pd

'------------------------------------------------ Data set import -------------------------------------------------'

taiex_df = get_TAIEX()
taiex = taiex_df.avg               
taiex = taiex.to_numpy()  

nasdaq_df = get_NASDAQ()
nasdaq = nasdaq_df.avg               
nasdaq = nasdaq.to_numpy()    

sp500_df = get_SP500()
sp500 = sp500_df.Avg               
sp500 = sp500[11500:16000]
sp500 = sp500.to_numpy()     


'------------------------------------------------ Gridsearch Parameters -------------------------------------------------'

datasets = [taiex]
dataset_names = ['taiex']
diff = 1                                       #If diff = 1, data is differentiated
partition_parameters = np.arange(1,11)         #partiions must be a list
orders = [1,2,3]
partitioners = ['SODA']                        #partitioners: 'chen' 'SODA' 'ADP' 'DBSCAN' 'CMEANS' 'entropy' 'FCM'  
mfs = ['triangular']                           #mfs: 'triangular' ou 'trapezoidal' ou 'gaussian'


'------------------------------------------------ Running the model -------------------------------------------------'


'Builds and runs the model'
run_sliding_window(datasets,dataset_names,diff,partition_parameters,orders,partitioners,mfs)


