# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:15:41 2020

Arquivo com as principais funções para sistemas fuzzy tipo-2
"""
import numpy as np
import pandas as pd
#from pandas import Dataframe
import matplotlib.pyplot as plt
from numpy import linspace
from SODA_T2FTS.SODA_T2FTS.T2FTS import FuzzySet, tri_mf, IT2FS_plot, min_t_norm, max_s_norm, TR_plot, crisp
from SODA_T2FTS.SODA_T2FTS.T2FTS import Type2Model

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt





        
def entropia(self,data,numero):
    
    tipo2.config_inicial(self,data,numero)
    
    from pyFTS.partitioners import Entropy
    partition = Entropy.EntropyPartitioner(data=data,npart=numero)

    conjuntos = partition.lista

    'Define o número de conjuntos'
    numero_de_sets = len(conjuntos)
    
    
    dominio_inf = conjuntos[0][0]
    dominio_sup = conjuntos[-1][2]
    domain = linspace(dominio_inf,dominio_sup,dominio_sup-dominio_inf)   #Gerao univeros de discurso
    
    dict_sets = {}   #Dicionário contendo os sets
    
    for x in range(1,numero_de_sets+1):
          
        r,t,y = conjuntos[x-1]
        b_esq = r        #Base esquerda
        topo_tri = t     #Topo do triangulo
        b_dir = y  
        
        fou_right = (b_dir-topo_tri)*0.4        #A mancha nao pode ser maior dos que os vertices do triangulo
        fou_left = (topo_tri-b_esq)*0.4         #Calcula a mancha da esquerda e direita e pega a menor para valer para os dois
        fou = min(fou_left,fou_right)
        
        dict_sets['A%d' %x] = IT2FS(domain,tri_mf, [b_esq, topo_tri, b_dir, 1],tri_mf, [b_esq+fou, topo_tri, b_dir-fou, 1])
    
    return dict_sets,numero_de_sets  
 


def conjuntos_soda_definindo_tamanho_dos_sets(self,data,data_original,gridsize):
    """
    INACABADA - NÃO USE
    
    """
    
    print("Gridsize:",gridsize )
    
    from pyT2FTS.SODA import SODA_function 
    
    soda_idx = SODA_function(data_original,gridsize)
            
    dados_idx = pd.DataFrame(soda_idx,columns=['idx'])
            
    'Pega valores minimos e maximos' 
    minimo = dados_idx['idx'].min()
    maximo = dados_idx['idx'].max()
    
    'Define o numero de sets'
    numero_de_sets = maximo
    
    'Encontra os indices em que cada linha pertence a cada conjunto'
    conjuntos = []
    for x in range (minimo,maximo+1):
        linhas = dados_idx[dados_idx['idx']==x].index.values.astype(int)
        conjuntos.append(linhas)  
        
        
    'Descobre qual o valor correpondente no vetor de dados a cada indice de cada conjunto'
    valores = []
    lista_aux = []
    for lim in conjuntos:
        for x in lim:  
            lista_aux.append(data[x])
        valores.append(lista_aux)
        lista_aux = []
            
    'Acha os limites dos conjuntos trinagulares ( b_esq e b_dir)'
    limites_dados = []
    for conjunto in valores:
        m = min(conjunto)   
        mx = max(conjunto)
        limites_dados.append(m)
        limites_dados.append(mx)
        
     
    'Funcao para exlcuir conjuntos que tenham dimensoes parecidas, para evitar conjuntos duplicados'   
    'Compara cada valor com os outros da lista para ver se estao proximos' 
    for x in range(0,len(limites_dados)-1,2):
        inf = limites_dados[x]
        sup = limites_dados[x+1]
        for y in range(len(limites_dados)-1):
            inf2 = limites_dados[y]
            sup2 = limites_dados[y+1]
            if inf > inf2 and (inf - inf2) <= 40 and sup > sup2 and (sup - sup2) <= 40:
                print("conjunto repetido")
                limites_dados[y] = limites_dados[x]
                limites_dados[y+1] = limites_dados[x+1]
        
    'Apaga valores repetidos'
    limites_dados = list(dict.fromkeys(limites_dados)) 
    
    
    numero_de_sets = int(len(limites_dados)/2)
    
    'configurações inciais'
    tipo2.config_inicial(self,data,numero_de_sets)
  
    'Geração dos sets'
    dict_sets = {}
    
    y = 1
    
    for x in range(0,len(limites_dados),2):
            
        b_esq = limites_dados[x]        #Base esquerda
        topo_tri = b_esq + (limites_dados[x+1] - limites_dados[x])/2
        b_dir = limites_dados[x+1]  
            
        fou_right = (b_dir-topo_tri)*0.4        #A mancha nao pode ser maior dos que os vertices do triangulo
        fou_left = (topo_tri-b_esq)*0.4 
        
                
        dict_sets['A%d' %y] = IT2FS(self.domain,tri_mf, [b_esq, topo_tri, b_dir, 1],tri_mf, [b_esq+fou_left, topo_tri, b_dir-fou_right, 1])
        y = y+1
                         
    return dict_sets
 

   






    
    
def agrupar_regras(n_conj,lista_regras):
    """First Order Conventional Fuzzy Logical Relationship Group
    
    Pode ser usada para conjuntos sobrepostos e sequenciais
  
    Faz o agrupamento de regras para o mesmo Left Hand Side (LHS).
    Printa as relações de regras fuzzy entre os conjuntos
    Separa todos os antecedentes e agrega os consequentes de cada um
    
    :params:           
    :n_conj: number of sets of the model
    :lista_regras: List containing the rules to be grouped
   
    Return
    
    :flrg: dicionario com os agrupamentos das regras. As keys sao os nomes
    dos LHSs e os values os RHSs respectivos.     
    
    OBS: Para sistemas de ordem maior que 1, o agrupamento é feito com base nos 
    conjuntos ativados apenas por x1, ignorando os outros inputs anteriores
    """
        
    numero_regras = len(lista_regras)

    print("Agrupamento de regras Fuzzy")
    print("Número de regras: ", numero_regras)
    
    flrg = {}    
    flrg_aux = []
    
    """Primeiro escaneia a lista de regras para escolher apenas os vetores com LHS
    igual a n (n = 1...n_conj) Depois pega os RHSs para este LHS e coloca em um
    dicionario.  """

    for LHS in range(1,n_conj+1):  #escaneia as regras para todas as regras com LHS igual ao numero do for
        selected = list(filter(lambda x: x[-2] == LHS, lista_regras))            
        for RHS in selected:        #Escaneia todas as regras selecionadas na etapa anterior
            rhs2 = RHS[-1]          #Pega o consequente da regra
            if rhs2 not in flrg_aux:    #Só adiciona o conjunto se ele for NOVO
                flrg_aux.append(rhs2)  
        if flrg_aux:                            #Verifica se a lista é vazia ou não, se nao for, faz o if
            flrg["A%d"%LHS] = flrg_aux
        flrg_aux = []
                   
    'Imprime os agrupamentos na tela' 

    for num in flrg:          
        print(num,"->", end = "")
        for x in flrg[num]:        
            print("A%d," %x , end =" ")
        print("")
        
    return flrg


 
  
def plot_forecast(teste_func,previsao):
    
    """Plota um gráfico dos dados_originas x previsão
    
    O gráfico correto tem como primeira amostra o teste = algumacoisa e previsao igual a None 
        ( a previsao eh feita para a proxima amostra)
    O gráfico correto tem como ultima amostra teste = None e previsao - algumacoisa,
        (a ultima previsao eh resultado do ultimo dado de teste)
    
    """
    'O ultimo dado de teste nao existe, por isso é atribuido a None'
    teste_func = teste_func.tolist()
    #teste_func.append(None)  # Último dado é só a previsão, nao existe valor na serie temporal
    
    #'O primeiro dado de previsao nao existe, por isso é atribuido a None'
    #previsao.insert(0, None)
  
    plt.figure()
    plt.figure(figsize=(20,10))
    x, = plt.plot(teste_func, label = "Série Original", color = 'b')
    y, = plt.plot(previsao, label = "Previsão Fuzzy Tipo-2", color = 'r')
    #plt.xticks(np.arange(1970, 1993,1)) 
    plt.legend(handles=[x, y])
    plt.show()  

    """'Salvando dados em excel'
    
    'tranforma listas em arrays numpy'
    if isinstance(teste_func, list):
        teste_func = np.array(teste_func)
    if isinstance(previsao, list):
        previsao = np.array(previsao)
        
    'Primeiro valor da previsao é zero'
    previsao = np.concatenate(([0],previsao))
  
    vect = pd.DataFrame.from_dict({'Teste':teste_func,'Previsao':previsao})
    vect.to_excel('vetores.xlsx', header=True, index=False)     """


def operador_regras_3entradas(conj_ativ_close,conj_ativ_lower,conj_ativ_high):
    
    """Finds the rules according to 1 type-1 observation and 2 type-2 observations
    
    Params
    Each input is a dictionary corresponding to other obervation (type-1 and type-2)
    To be applied as input to this function, the rules should have been already extracted
    using the function extract_rules() from the pyIT2FLS file.
    
    :conj_ativ_close: dictionary where the keys are the samples and the values are the activated sets
    """
    regras = []
    for x in range(1,len(conj_ativ_close)):
        'Cada value dos dicionarios sao uma lista de conjuntos, por isso tem que usar o [0]'
        tup = (conj_ativ_close["%d"%x][0],conj_ativ_lower["%d"%x][0],conj_ativ_high["%d"%x][0],conj_ativ_close["%d"%(x+1)][0])
        if tup not in regras:
            regras.append(tup)
    
    return regras



def operador_intersecao_uniao(n_amostras,conj_ativ_close,conj_ativ_lower,conj_ativ_high,flrg_close,flrg_lower,flrg_high):
    
    """
    Executa as operações de interseção e união das regras das séries temporais
    Pode ser usados para 1 variavel tipo-1 e 2 variaveis tipo-2 (total = 3 variaveis)
    To be applied as input to this function, the rules should have already been extracted
    using the function extract_rules() from the pyIT2FLS file. Then each list of rules
    should have been already grouped uding the function agrupar_regras().
    
    Return
    
    :regras_gerais_int: lista com regras da interseção em tuple (antecx1,antecx2,antecx3,conseq)
    :dict_int: dicionario em que keys sao as amostras (n° da amostra temporal) e values sao os conjuntos ativados por cada amostra
    :regras_gerais_union: lista com regras da uniao em tuple (antecx1,antecx2,antecx3,conseq)
    :dict_union dicionario em que keys sao as amostras (n° da amostra temporal) e values sao os conjuntos ativados por cada amostra
    
    """
    dict_int = {}
    dict_union = {}
    regras_gerais_int = []
    regras_gerais_union = []
    
    'Definindo quais conjuntos ativados a cada amostra para cada serie temporal'

    for x in range(1,n_amostras+1):
        
        LHS_close = conj_ativ_close['%d'%x]
        LHS_low = conj_ativ_lower['%d'%x]
        LHS_high= conj_ativ_high['%d'%x]
                    
        'Pega os consequentes de cada antecedente ativado no dia da amostra x'
        RHS_close = flrg_close['A%d'%LHS_close[0]] 
        RHS_lower = flrg_lower['A%d'%LHS_low[0]]
        RHS_high = flrg_high['A%d'%LHS_high[0]]
        
        'Faz a interseção dos valores ativados'
        intersection = list(set(RHS_close).intersection(set(RHS_lower),set(RHS_high)))                            
        if not intersection:    #Checa se interseção é vazia
            intersection = [LHS_close]   # Caso nao tenha interseção, usa LHS da amostra tipo-1
        
        dict_int['%d'%x] = intersection  #dicionario para guardar a interseção de cada amostra
        for i in intersection:              #Constroi a lista de regras gerais para serem adicionadas
            aux = (LHS_close[0],LHS_low[0],LHS_high[0],i)                
            if aux not in regras_gerais_int:
                regras_gerais_int.append(aux)
                                                           
        'Faz a união dos valores ativados'
        union = list(set(RHS_close).union(set(RHS_lower),set(RHS_high)))                            
        if not union:    #Checa se interseção é vazia
            union = [LHS_close]   # Caso nao tenha interseção, usa LHS da amostra tipo-1
        
        dict_union['%d'%x] = union
        for i in union:              #Constroi a lista de regras gerais para serem adicionadas
            aux = (LHS_close,LHS_low,LHS_high,i)                
            if aux not in regras_gerais_union:
                regras_gerais_union.append(aux)
    
    
    return regras_gerais_int,dict_int,regras_gerais_union,dict_union


def error_metrics(teste,previsao): 
    """
    Calcula metricas de erro para a previsao. 
    
    Teste e previsao devem estar em suas formas 'naturais', sem valores vazios.
    
    :params:
        
    :teste: lista com valores de teste SEM o primeiro dado (não tem correspondente)
    :previsao: Lista com valores previstos SEM o último dado (não tem correspondente)
         
    
    """
    'tranforma listas em arrays numpy'
    if isinstance(teste, list):
        teste = np.array(teste)
    if isinstance(previsao, list):
        previsao = np.array(previsao)
        
    udetheil = 0
  
    print("Error Metrics:")
                   
    mape = mape_function(teste, previsao)
    #print("MAPE", mape)
      
    mse = mean_squared_error(teste, previsao)
    #print("MSE:", mse)

    rmse = sqrt(mse)
    print("RMSE:", rmse)
    
    mae = mean_absolute_error(teste, previsao)  
    #print("MAE:", mae)
    
    ndei = rmse/np.std(previsao)   #O NDEI é rmse dividido pelo desvio padrao
    #print("NDEI: ", ndei)  
    
    lista_erros = [udetheil,mape,mse,rmse,mae,ndei]
            
    return lista_erros




def teste_ADF(dataset_names,datasets):
    
    """
    Performs the Augmented Dickey-Fuller test (ADF Test)
    H0 hypothesis: null hypothesis: series IS NOT stationaty
    H1 hypothesis: series IS stationaty
    
    If p > 0.05, accepted H0 ( time series is not stationary)
    If p <=0.05, reject H0   ( time series is stationary)
    
    Also:      
    If ADF Statistic < critical value 1% or 5% or 10%, that is the probability the series
    is stationary ( thus we can reject H0)
    
    Also:
    The more negative the ADF statistic, the stronger the evidence for rejecting the null hypothesis

     
    :params:
        dataset_names: list containing the dataset names
        datasets: list of datasets to be tested
        OBS: datasets and dataset names must be in the same order in both lists
    :return:
        resultado: Dataframe where values are dataset names and items are ADF statistic results
    """
    from statsmodels.tsa.stattools import adfuller
    
    dataset_number = 0
    resultado_ADF = []
    
    for data in datasets:
    
        resultado = []
    
        result = adfuller(data) 
        resultado.append(dataset_names[dataset_number])          
        resultado.append(result[0])
        resultado.append(result[1])
        resultado.append(result[4]['1%'])
        resultado.append(result[4]['5%'])
        resultado.append(result[4]['10%'])

        resultado_ADF.append(resultado)
                
        dataset_number = dataset_number + 1
        
    'passa para dataframe'
    df = pd.DataFrame(resultado_ADF, columns = ['Dataset','Statistic','p-value','Cr.Val. 1%','Cr.Val. 5%','Cr.Val. 10%'])
      
    return df

def metricas_erro_antiga(teste,previsao): 
    """
    Calcula metricas de erro para a previsao. 
    
    Teste e previsao devem estar em suas formas 'naturais', sem valores vazios.
    
    
    Do vetor teste, devemos pegar todos os valores menos o primeiro, que nao tem correspondente
    Do vetor previsao, pegar todos os valores menos o ultimo, que nao tem correspondente
    
    """
    'tranforma listas em arrays numpy'
    if isinstance(teste, list):
        teste = np.array(teste)
    if isinstance(previsao, list):
        previsao = np.array(previsao)
        
    'elimina a ultima previsao, que nao sera usada'
    previsao = np.delete(previsao, -1)
    
    print("Metricas de erro:")
            
    udetheil = udetheil_statistic(teste,previsao)
    print("U de Theil: ", udetheil)
            
    'Exclui o primeiro valor de teste, que nao tem correspondente na previsao'
    teste = np.delete(teste, 0)
    
    mape = mape_function(teste, previsao)
    print("MAPE", mape)
      
    mse = mean_squared_error(teste, previsao)
    print("MSE:", mse)

    rmse = sqrt(mse)
    print("RMSE:", rmse)
    
    mae = mean_absolute_error(teste, previsao)  
    print("MAE:", mae)
    
    ndei = rmse/np.std(previsao)   #O NDEI é rmse dividido pelo desvio padrao
    print("NDEI: ", ndei)  
    
    lista_erros = [udetheil,mape,mse,rmse,mae,ndei]
            
    return lista_erros


    
    
def udetheil_statistic(teste,previsao):
            
    'U de Theil'
    lista_numerador = []
    lista_denominador = []
          
    'adiciona um valor no inicio da previsao apenas para facilitar o algoritmo abaixo'
    previsao = np.concatenate(([0],previsao))
          
    'Constroi o numerador e denominador da formula de u de theil'
    for x in range(1,len(previsao)-1):
        numerador = ((previsao[x]-teste[x])/teste[x-1])**2  #Ao quadrado
        lista_numerador.append(numerador) 
        denominador = ((teste[x]-teste[x-1])/teste[x-1])**2
        lista_denominador.append(denominador) 
        
    udetheil_numerador = sum(lista_numerador)          
    udetheil_denominador = sum(lista_denominador)
    
    udetheil = sqrt(udetheil_numerador/udetheil_denominador)
    
    return udetheil
   

def theil_inequality(teste,previsao):
            
    'U de Theil'
    lista_numerador = []
    lista1 = []
    lista2 = []
    
    for x in range(1,len(previsao)-1):
        numerador = (teste[x]-previsao[x])**2  #Ao quadrado
        lista_numerador.append(numerador) 
                
        denom1 = teste[x]**2
        lista1.append(denom1)
        denom2 = previsao[x]**2
        lista2.append(denom2)

    udetheil_numerador = sum(lista_numerador)
    udetheil_numerador = udetheil_numerador/len(teste)
    udetheil_numerador = sqrt(udetheil_numerador)
    
    denom1 = sum(lista1)/len(teste)
    denom1 = sqrt(denom1)
    denom2 = sum(lista2)/len(previsao)
    denom2 = sqrt(denom2)
    
    udetheil_denominador = denom1 + denom2
  
    udetheil = udetheil_numerador/udetheil_denominador       
    print("Theil inequality:",udetheil)
    
    return udetheil

def mape_function(teste,previsao):
    
    if isinstance(teste, list):
        teste = np.array(teste)
    if isinstance(previsao, list):
        previsao = np.array(previsao)
    return np.nanmean(np.abs(np.divide(np.subtract(teste, previsao), teste))) * 100



def plot_dataset(dataset,xlabel,ylabel):
    """
    Use this function to plot a dataset for papers
    
    """
    
    plt.figure()
    plt.figure(figsize=(16,6))
    plt.plot(dataset)
    plt.yticks(fontsize=16) 
    plt.xticks(fontsize=16)
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel(ylabel,fontsize=20)
    #plt.axvline(x=4208, color = "red")
    plt.savefig("dataset_plotted.png", format="png", dpi=300, bbox_inches="tight")
    
    
def plot_partitions(dataset):
    
    
    
    datasets = [dataset]
    dataset_names = ['DATASET']
    diff = 1                                   #Se diff = 1, diferencia os dados. Se diff = 0, não diferencia
    particoes = np.arange(8,11)                 #particoes deve ser uma lista
    ordens = [2]
    partitioners = ['FCM']            #partitioners: 'chen' 'SODA' 'ADP' 'DBSCAN' 'CMEANS' 'entropy' 'FCM'  
    mfs = ['triangular']         #mfs: 'triangular' ou 'trapezoidal' ou 'gaussian'
    
    '------------------------------------------------ Running the model -------------------------------------------------'
    
    
    'Builds and runs the model'
    #janela_deslizante(datasets,dataset_names,diff,particoes,ordens,partitioners,mfs)

