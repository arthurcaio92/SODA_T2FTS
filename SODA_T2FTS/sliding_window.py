# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from SODA_T2FTS.SODA_T2FTS.Headquarters import T2FTS
import pickle #To save the data as the process goes
import time

    
def run_sliding_window(datasets,dataset_names,diff,partition_parameters,orders,partitioners,mfs):
    
    """
    
    Performs sliding window methodology over a time series
    
    params:
    :datasets: List of time series
    :dataset_names: List of time series names
    :diff: List of flag to differentiate or not the input data
    :partition_parameters: List of partition parameters to be used. If method_part is SODA or ADP, then partitions is the gridsize.
    :orders: List of List of model orders ( n. or lags in forecasting)
    :partitioners: List of partitioners to be used
    :mfs: List of membership functions to be used
    
    Return (currently disabled)
    
    :df_general: dataframe containing general error, about each window    
    :df_specific: dataframe with average errors accounting for all window  
    Saves an excel file (.xlsx) in the end with error metrics

    """
    
    'Checks if the right number and datasets and their names was provided'
    if len(datasets) != len(dataset_names):
        raise Exception("Please specify the correct number of datasets and their names")
        
    
    'Auxiliar variable to know the name of the data set'
    name_index = 0
    
    for data in datasets:
        
        data_name = dataset_names[name_index]
        name_index = name_index + 1 #update for the nex loop
        
        
        for method_part in partitioners:
            
            for mf_type in mfs:
           
                'Verifications'
                
                'If any of these situations happen, ignore this loop and goes to the next'
                
                if method_part == 'CMEANS' and (mf_type == 'trapezoidal' or mf_type == 'gaussian'):
                    print("-------------\n","WARNING: ", method_part," does not support ",mf_type," membership function","\n-------------",)
                    continue
                
                if (method_part == 'FCM' or method_part == 'entropy') and (mf_type == 'gaussian'):
                    print(method_part," does not support ",mf_type," membership function")
                    continue
                
                'Let''s measure the total elapsed time for the whole process to be completed'
                start_time = time.time()
    
                'Indicates the percentage of the windowsize to move the window'
                increment = 0.2  
            
                'list to sabe the errors'
                lista_rmse = []
                lista_partitions = []
                lista_rules = []
                lista_flrg = []
                
                general_errors = {'Gridsize':[],
                               'Partitions':[],
                               'Order':[],
                               'Window':[],
                               'UDETHEIL':[],
                               'MAPE':[],
                               'MSE':[],
                               'RMSE':[],
                               'MAE':[],
                               'NDEI':[],
                               'Avg_RMSE':[],
                               'Std_dev_RMSE':[],
                               'FLR':[],
                               'FLRG':[]
                              
                               }
                
                specific_errors = {'Gridsize':[],
                                    'Order':[],
                                    'Partitions': [],
                                    'mean_RMSE': [],
                                    'std_RMSE': [],
                                    'FLR':[],
                                    'FLRG':[],
                                    'Time(s)': [],
                                    'Total Time(s)': []
                                    
                                     }  
                
                
                '-----Begins the Gridsearch------'
                
                for part_param in partition_parameters:
                    
                    gridsize = part_param
                    
                    for lag in orders:
                    
                        '------------------------------------------------ Sliding Window -------------------------------------------------'
                           
                        #window_size = int(0.2*len(data))      #Tamanho da janela deslizante
                        window_size = 1000
                        
                        window_inf = 0
                        window_sup = window_size
                        
                        'Let''s measure the METHOD elapsed time '
                        method_start_time = time.time()
                        
                        
                        while (window_sup <= len(data)):
                            
                            dados = data[window_inf:window_sup]
                            print("Window: [", window_inf, ":",window_sup,"]")
                            print("MF:",mf_type)
                        
                            '------------------------------------------------ Window setup -------------------------------------------------'
                
                            'Define model order'
                            order = lag
                            
                                            
                            lista_erros,n_sets,FLR,FLRG = T2FTS(dados,method_part,mf_type,part_param,order=order,diff=diff)
                           
                            print("---------------------------------")
                               
                            '------------------------------------------------  Error Metrics  ------------------------------------------'
                            'Gets the RMSE from the errors list'
                            lista_rmse.append(lista_erros[3])
                            
                            'Adds the number of rules to the respective list'
                            lista_rules.append(FLR)
                            lista_flrg.append(FLRG)
                            lista_partitions.append(n_sets)
                            
                            'Builds the general_errors dictionary with data for each window'
                            general_errors['Gridsize'].append(gridsize)                
                            general_errors['Partitions'].append(n_sets)
                            general_errors['Order'].append(order)
                            general_errors['Window'].append("{}:{}".format(window_inf,window_sup))
            
                            general_errors['UDETHEIL'].append(lista_erros[0])
                            general_errors['MAPE'].append(lista_erros[1])
                            general_errors['MSE'].append(lista_erros[2])
                            general_errors['RMSE'].append(lista_erros[3])
                            general_errors['MAE'].append(lista_erros[4])                
                            general_errors['NDEI'].append(lista_erros[5]) 
                            general_errors['Avg_RMSE'].append(None)   
                            general_errors['Std_dev_RMSE'].append(None)   
                            general_errors['FLR'].append(FLR)
                            general_errors['FLRG'].append(FLRG)
                           
                            'Slides the window'
                            window_inf = window_inf+200
                            window_sup = window_sup+200
                            
                            #window_inf = int(window_inf + window_size * increment)
                            #window_sup = int(window_sup + window_size * increment)
                            
                        
                        'Ends time measurement'
                        method_end_time = time.time()
                        
                        method_elapsed_time = method_end_time - method_start_time
                        
                        'Calculates RMSE average and std. dev. for all windows'
                        
                        avg_rmse = np.mean(lista_rmse)
                        std_rmse = np.std(lista_rmse)
                        avg_partitions = np.mean(lista_partitions)
                        avg_rules = np.mean(lista_rules)
                        avg_flrg = np.mean(lista_flrg)
            
                        
                        'Fills the lines correspondent to the mean values of the metrics. The other lines are zero'
                        'Some are None because it has to show nothing on the Excel file'
                        
                        '### First line: empty ###'
                        general_errors['Gridsize'].append(None)
                        general_errors['Partitions'].append(None)
                        general_errors['Order'].append(None)
                        general_errors['Window'].append(None)
                   
                        general_errors['UDETHEIL'].append(None)
                        general_errors['MAPE'].append(None)
                        general_errors['MSE'].append(None)
                        general_errors['RMSE'].append(None)
                        general_errors['MAE'].append(None)                
                        general_errors['NDEI'].append(None)   
                        general_errors['Avg_RMSE'].append(None) 
                        general_errors['Std_dev_RMSE'].append(None)  
                        general_errors['FLR'].append(None)
                        general_errors['FLRG'].append(None)
                        
                        '### Second line: Error metrics averages ###'
                        general_errors['Gridsize'].append('Médias:')
                        general_errors['Partitions'].append(avg_partitions)
                        general_errors['Order'].append(None)
                        general_errors['Window'].append(None)
                   
                        general_errors['UDETHEIL'].append(None)
                        general_errors['MAPE'].append(None)
                        general_errors['MSE'].append(None)
                        general_errors['RMSE'].append(None)
                        general_errors['MAE'].append(None)                
                        general_errors['NDEI'].append(None)   
                        general_errors['Avg_RMSE'].append(avg_rmse) 
                        general_errors['Std_dev_RMSE'].append(std_rmse)  
                        general_errors['FLR'].append(avg_rules)
                        general_errors['FLRG'].append(avg_flrg)
            
                        '### Third line: empty ###'
                        general_errors['Gridsize'].append(None)
                        general_errors['Partitions'].append(None)
                        general_errors['Order'].append(None)
                        general_errors['Window'].append(None)
                   
                        general_errors['UDETHEIL'].append(None)
                        general_errors['MAPE'].append(None)
                        general_errors['MSE'].append(None)
                        general_errors['RMSE'].append(None)
                        general_errors['MAE'].append(None)                
                        general_errors['NDEI'].append(None)   
                        general_errors['Avg_RMSE'].append(None) 
                        general_errors['Std_dev_RMSE'].append(None)  
                        general_errors['FLR'].append(None)
                        general_errors['FLRG'].append(None)
                        
                        'Builds the specific_error dictionary'   
                                              
                        specific_errors['Gridsize'].append(gridsize)   
                        specific_errors['Order'].append(order)         
                        specific_errors['Partitions'].append(avg_partitions) 
                        specific_errors['FLR'].append(avg_rules)  
                        specific_errors['FLRG'].append(avg_flrg)  
                        specific_errors['mean_RMSE'].append(avg_rmse)
                        specific_errors['std_RMSE'].append(std_rmse)
                        specific_errors['Time(s)'].append(method_elapsed_time)
                        specific_errors['Total Time(s)'].append(None)
            
                        
                        'Prints the results'
                        if method_part == 'chen':
                            r = "RMSE avg - part: " + str(part_param) + ", Order: " + str(order)
                            print("[",r,"]:",avg_rmse)
                            print("---------------------------------")
            
                        elif method_part == 'SODA' or method_part == 'ADP': 
                            r = "RMSE avg - Gridsize: " + str(gridsize) + ", Order: " + str(order)
                            print("[",r,"]:",avg_rmse)
                            print("---------------------------------")
                        
                        else:
                            r = "RMSE avg - Parâmetro: " + str(gridsize) + ", Order: " + str(order)
                            print("[",r,"]:",avg_rmse)
                            print("---------------------------------")
                        
                            
                                        
                        'Use pickle to save the dicts after each window as backup'
                        
                        pickle_out = open("general.pickle","wb")
                        pickle.dump(general_errors, pickle_out)
                        pickle_out = open("specific.pickle","wb")          
                        pickle.dump(specific_errors, pickle_out)         
                        pickle_out.close()
                    
                        'Resets the lists'
                        lista_rmse = []  
                        lista_rules = []
                        lista_flrg = []
                        lista_partitions = []
                        
                
                'Ends time measurement'
                end_time = time.time()      
                total_elapsed_time = end_time - start_time
                
                'Adds the final line with the total elapsed time'
                specific_errors['Gridsize'].append(None)   
                specific_errors['Order'].append(None)         
                specific_errors['Partitions'].append(None) 
                specific_errors['FLR'].append(None)  
                specific_errors['FLRG'].append(None)  
                specific_errors['mean_RMSE'].append(None)
                specific_errors['std_RMSE'].append(None)
                specific_errors['Time(s)'].append('Total Elapsed Time:')
                specific_errors['Total Time(s)'].append(total_elapsed_time)
                           
                
                '------------------------------------------------  Save to excel  ------------------------------------------'
            
                'Defines the name of the final file'
                if diff == 0:  
                    name_file = method_part + "_semdiff_" + data_name + "_" + mf_type + "_" + str(partition_parameters[0]) + "a" + str(partition_parameters[-1]) + ".xlsx"
                    
                elif diff == 1:   
                    name_file = method_part + "_diff_" + data_name + "_" + mf_type + "_" + str(partition_parameters[0]) + "a" + str(partition_parameters[-1]) + ".xlsx"      
                       
                
                print("Saved file:",name_file)
                writer = pd.ExcelWriter(name_file, engine='xlsxwriter')
                        
                df_general = pd.DataFrame(data=general_errors)
                df_specific = pd.DataFrame(data=specific_errors)
                #df_especifico.columns = ['Gridsize','Partitions','RMSE medio 1_Order', 'Desvio padrao RMSE 1_Order','RMSE medio 2_Order', 'Desvio padrao RMSE 2_Order','RMSE medio 3_Order', 'Desvio padrao RMSE 3_Order','FLR','FLRG']    
                       
                df_general.to_excel(writer, sheet_name='General errors',index = False)
                df_specific.to_excel(writer, sheet_name='Especific errors',index = False)
               
                writer.save()
                
                #Downloads the Excel file to computer
                #from google.colab import files
                #files.download(name_file)
                
                
             
                
        
        

