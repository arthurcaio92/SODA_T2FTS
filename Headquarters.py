# -*- coding: utf-8 -*-
from pyT2FTS.T2FTS import Type2Model,IT2FS_plot
from pyT2FTS.Tools import error_metrics,plot_forecast 
from pyT2FTS.Partitioners import SODA_part,ADP_part,DBSCAN_part,FCM_part,ENTROPY_part,CMEANS_part,HUARNG_part
from pyT2FTS.Transformations import Differential
import numpy as np


"""
    Function that trains and tests a time series called 'data'

    :params:
    :data: data to be trained and tested
    :diff: flag to differentiate or not the input data
    :order: model order ( n. or lags in forecasting)
    :number_of_sets: n. of fuzzy sets created 
    
"""
    
def T2FTS(data,method_part,mf_type,partition_parameters,order,diff):
        
    '------------------------------------------------ Setup ------------------------------------------'
    
    'Training takes 80% of the data'
    training_interval = int(0.8 * len(data))         
     
    training_data = data[:training_interval]
  
    'Testing takes the remaining 20%'
    test_data = data[training_interval:]
          
    'Checks if the data must be differentiated'
    if diff == True:
        training_data_orig = training_data
        test_data_orig = test_data
    
        tdiff = Differential(1) 
        training_data = tdiff.apply(training_data_orig)
        test_data = tdiff.apply(test_data_orig)
    
    'Create an object of the class Type2Model'
    modelo = Type2Model(training_data,order) 
    
        
    '------------------------------------------------ Fuzzy sets generation  -------------------------------------------------'

    if method_part == 'chen':
        number_of_sets = partition_parameters
        modelo.grid_partitioning(partition_parameters, mf_type)
        
    elif method_part == 'SODA':
        gridsize = partition_parameters
        number_of_sets = SODA_part(training_data,gridsize)
        modelo.grid_partitioning(number_of_sets, mf_type)   
        
    elif method_part == 'ADP':
        gridsize = partition_parameters
        number_of_sets = ADP_part(training_data, gridsize)
        modelo.grid_partitioning(number_of_sets, mf_type)
        
    elif method_part == 'DBSCAN':
        eps = partition_parameters
        number_of_sets = DBSCAN_part(training_data, eps)
        modelo.grid_partitioning(number_of_sets, mf_type)
        
    elif method_part == 'CMEANS': 
        k = partition_parameters
        cmeans_params = CMEANS_part(training_data, k, mf_type)
        number_of_sets = len(cmeans_params)
        modelo.generate_uneven_length_mfs(number_of_sets, mf_type, cmeans_params)
    
    elif method_part == 'entropy':
        k = partition_parameters
        entropy_params = ENTROPY_part(training_data,k, mf_type)
        number_of_sets = len(entropy_params)
        modelo.generate_uneven_length_mfs(number_of_sets, mf_type, entropy_params)
        
    elif method_part == 'FCM':
        k = partition_parameters
        fcm_params = FCM_part(training_data,k, mf_type)
        number_of_sets = len(fcm_params)
        modelo.generate_uneven_length_mfs(number_of_sets, mf_type, fcm_params)
    
    elif method_part == 'huarng':
        huarng_params = HUARNG_part(training_data)
        number_of_sets = len(huarng_params)
        modelo.generate_uneven_length_mfs(number_of_sets,huarng_params)
        
    else:
        raise Exception("Method %s not implemented" % method_part)
        
        
    #Plot partition graphs
    #plot_title = str(number_of_sets) + ' partitions'
    #IT2FS_plot(*modelo.dict_sets.values(),title= plot_title)
    
    '------------------------------------------------ Training  ------------------------------------------'
        
    'Treina o modelo'
    FLR,FLRG = modelo.training()
    
    '------------------------------------------------  Testing  ------------------------------------------'
    'Clips the test data for them to be inside the Universe of Discourse'
    test_data = np.clip(test_data, modelo.dominio_inf+1, modelo.dominio_sup-1)

    
    print("Partitioner:",method_part,"| N. of sets:", number_of_sets, "| Order:", order)
    print("")
    forecast_result = modelo.predict(test_data)   

    
    'Return values to original scale (i.e. undo the diff)'
    if diff == True:
        forecast_result = forecast_result[1:] #faz isso por causa da diferenciação
        forecast_result = tdiff.inverse(forecast_result,test_data_orig)
        test_data = test_data_orig[order:]  # Para plotar e metricas de erro deve usar a serie original
        
    else:       
        test_data = test_data[order:]  # O primeiro item nao tem correspondente na previsao
        forecast_result = forecast_result[:-1]

    '------------------------------------------------  Métricas de erro  ------------------------------------------'
    error_list = error_metrics(test_data,forecast_result)
        
    'Plots forecast graph data x forecast'      
    #plot_forecast(test_data,forecast_result)
    
    
    return error_list,number_of_sets,FLR,FLRG
    
    
