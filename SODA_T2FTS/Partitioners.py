# -*- coding: utf-8 -*-
from pandas import DataFrame

  
    

def SODA_part(data,gridsize):
    
    """Retorna apenas o numero de conjuntos encontrado pelo SODA"""
    
    print("Gridsize:",gridsize)
    
    from SODA_T2FTS.SODA import SODA_function 
    
    soda_idx = SODA_function(data,gridsize)
                
    dados_idx = DataFrame(soda_idx,columns=['idx'])
            
    maximo = dados_idx['idx'].max()
    
    'Define o numero de sets'
    numero_de_sets = maximo
                         
    return numero_de_sets
    
 
def ADP_part(data,gridsize, distancetype='chebyshev'):
    'A distancia pode ser chebyshev, euclidean, cityblock, sqeuclidean ou cosine'
	
    """Retorna apenas o numero de conjuntos encontrado pelo ADP"""
    
    print("Gridsize: {}".format(gridsize))
    
    from SODA_T2FTS.OfflineADP import ADP 
    #from OfflineADP import ADP 
    
	#Make it a two-column dataframe       
    dados = DataFrame(data, columns = ['avg'])
    dados.insert(0, '#', range(1,len(dados)+1))
    
    centre, idx = ADP(dados,gridsize)           
    
    'Define o numero de sets'
    numero_de_sets = len(centre)
                         
    return numero_de_sets


def DBSCAN_part(data, eps):
    from sklearn.cluster import DBSCAN
    
    print("Parâmetro:",eps )
    
    #Transforma a base de treinos em um array 2D
    dados = DataFrame(data, columns = ['avg'])
    dados.insert(0, '#', range(1,len(dados)+1))
    dados = dados.to_numpy()

    #Executa o DBSCAN
    db = DBSCAN(eps = eps).fit(dados)
    
    #Salva as labels do modelo em uma variável
    labels = db.labels_

    #Conta o número de labels ignorando o -1, se presente, e salva numa variável
    numero_de_sets = len(set(labels)) - (1 if -1 in labels else 0)
    
    'Identifica a quantidade de outliers'
    n_noise = list(labels).count(-1)
    'A pocentagem de noise/total'
    r = n_noise/len(dados)
    
    
    print(f'EPS: {eps}')
    print(f'Noise points: {n_noise} ({r*100}%)')
    
    
    #Caso o algoritmo não crie clusters, retornar 1 para evitar problemas futuros
    if numero_de_sets == 0:
        return 1
    
    return numero_de_sets


def CMEANS_part(data, k, mf_type):
    from pyFTS.partitioners import CMeans
    from pyFTS.common import Membership #utilizado no argumento func
    
    print("Parâmetro:",k )
    
    if mf_type == 'triangular':
        mf = Membership.trimf
    if mf_type == 'trapezoidal':
        mf = Membership.trapmf
    
    #Executa o particionamento e salva num objeto
    obj = CMeans.CMeansPartitioner(data=data, npart=k, func=mf)
    
    ##Lista para guardar os parâmetros da função de cada set    
    cmeans_params = []
    for i in range(1,k+1):
        cmeans_params.append(obj.sets['A'+str(i)].parameters)
    
    return cmeans_params

def ENTROPY_part(data, k, mf_type):
    from pyFTS.partitioners import Entropy
    from pyFTS.common import Membership
    
    print("Parâmetro:",k )
    
    if mf_type == 'triangular':
        mf_type = Membership.trimf
    elif mf_type == 'trapezoidal':
        mf_type = Membership.trapmf

    #Executa o particionamento e salva num objeto
    obj = Entropy.EntropyPartitioner(data=data, npart=k, func=mf_type)
    
    ##Lista para guardar os parâmetros da função de cada set
    entropy_params = []
    for i in range(0,len(obj.sets)):
        entropy_params.append(obj.sets['A'+str(i)].parameters)
    
    return entropy_params

def FCM_part(data,k, mf_type):
    from pyFTS.partitioners import FCM
    from pyFTS.common import Membership
    
    print("Parâmetro:",k )
    
    if mf_type == 'triangular':
        mf_type = Membership.trimf
    elif mf_type == 'trapezoidal':
        mf_type = Membership.trapmf
    
    #Executa o particionamento e salva num objeto
    obj = FCM.FCMPartitioner(data=data, npart=k,func=mf_type)   
    
    ##Lista para guardar os parâmetros da função de cada set
    fcm_params = []
    for i in range(1,len(obj.sets)+1):
        fcm_params.append(obj.sets['A'+str(i)].parameters)
    
    return fcm_params

def HUARNG_part(data):
    from pyFTS.partitioners import Huarng
    
    #Executa o particionamento e salva num objeto
    obj = Huarng.HuarngPartitioner(data=data)
    
    #Lista para dict keys
    keys = []
    for i in range(1,len(obj.sets)+1):
        keys.append('A'+str(i))
    
    ##Lista para guardar os parâmetros (a, b, c) da função triangular de cada set
    huarng_params = []
    for i in keys:
        huarng_params.append(obj.sets[i].parameters)
    
    return huarng_params