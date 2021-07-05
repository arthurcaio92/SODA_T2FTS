# -*- coding: utf-8 -*-
"""
Distancias
euclidean - linha reta entre os pontos
mahalanobis - correlacao entre as variaveis (determina similaridade)
cityblock - distancia das projecoes dos pontos (taxicab/manhattan)
chebyshev - maior distancia entre as coordenadas (rei)
minkowski - generalizacao de outras distancias:
    p = 1  -  cityblock,
    p = 2  -  euclidean,
    p = infinite - chebyshev.
canberra - versao com pesos da cityblock, sensivel para pontos proximos a origem

"""
    
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd


import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist, squareform
    
    
def grid_set(data, N):
    _ , W = data.shape
    AvD1 = data.mean(0)
    X1 = np.mean(np.sum(np.power(data,2),axis=1))
    grid_trad = np.sqrt(2*(X1 - AvD1*AvD1.T))/N
    Xnorm = np.sqrt(np.sum(np.power(data,2),axis=1))
    aux = Xnorm
    for i in range(W-1):
        aux = np.insert(aux,0,Xnorm.T,axis=1)
    data = data / aux
    seq = np.argwhere(np.isnan(data))
    if tuple(seq[::]): data[tuple(seq[::])] = 1
    AvD2 = data.mean(0)
    grid_angl = np.sqrt(1-AvD2*AvD2.T)/N
    return X1, AvD1, AvD2, grid_trad, grid_angl


def pi_calculator(Uniquesample, mode):
    UN, W = Uniquesample.shape
    if mode == 'euclidean' or mode == 'mahalanobis' or mode == 'cityblock' or mode == 'chebyshev' or mode == 'canberra':
        AA1 = Uniquesample.mean(0)
        X1 = sum(sum(np.power(Uniquesample,2)))/UN
        DT1 = X1 - sum(np.power(AA1,2))
        aux = []
        for i in range(UN): aux.append(AA1)
        aux2 = [Uniquesample[i]-aux[i] for i in range(UN)]
        uspi = np.sum(np.power(aux2,2),axis=1)+DT1
    
    if mode == 'minkowski':
        AA1 = Uniquesample.mean(0)
        X1 = sum(sum(np.power(Uniquesample,2)))/UN
        DT1 = X1 - sum(np.power(AA1,2))
        aux = np.matrix(AA1)
        for i in range(UN-1): aux = np.insert(aux,0,AA1,axis=0)
        aux = np.array(aux)
        uspi = np.sum(np.power(cdist(Uniquesample, aux, mode, p=1.5),2),1)+DT1
    
    if mode == 'cosine':
        Xnorm = np.matrix(np.sqrt(np.sum(np.power(Uniquesample,2),axis=1))).T
        aux2 = Xnorm
        for i in range(W-1):
            aux2 = np.insert(aux2,0,Xnorm.T,axis=1)
        Uniquesample1 = Uniquesample / aux2
        AA2 = np.mean(Uniquesample1,0)
        X2 = 1
        DT2 = X2 - np.sum(np.power(AA2,2))
        aux = []
        for i in range(UN): aux.append(AA2)
        aux2 = [Uniquesample1[i]-aux[i] for i in range(UN)]
        uspi = np.sum(np.sum(np.power(aux2,2),axis=1),axis=1)+DT2
        
    return uspi


def Globaldensity_Calculator(data, distancetype):
    
    Uniquesample, J, K = np.unique(data, axis=0, return_index=True, return_inverse=True)
    Frequency, _ = np.histogram(K,bins=len(J))
    uspi1 = pi_calculator(Uniquesample, distancetype)
    sum_uspi1 = sum(uspi1)
    Density_1 = uspi1 / sum_uspi1
    uspi2 = pi_calculator(Uniquesample, 'cosine')
    sum_uspi2 = sum(uspi2)
    Density_2 = uspi1 / sum_uspi2
    
    GD = (Density_2+Density_1) * Frequency

    index = GD.argsort()[::-1]
    GD = GD[index]
    Uniquesample = Uniquesample[index]
    Frequency = Frequency[index]
 
    return GD, Uniquesample, Frequency


def chessboard_division(Uniquesample, MMtypicality, interval1, interval2, distancetype):
    L, W = Uniquesample.shape
    if distancetype == 'euclidean':
        W = 1
    BOX = [Uniquesample[k] for k in range(W)]
    BOX_miu = [Uniquesample[k] for k in range(W)]
    BOX_S = [1]*W
    BOX_X = [sum(Uniquesample[k]**2) for k in range(W)]
    NB = W
    BOXMT = [MMtypicality[k] for k in range(W)]
    
    for i in range(W,L):
        if distancetype == 'minkowski':
            a = cdist(Uniquesample[i].reshape(1,-1), BOX_miu, metric=distancetype, p=1.5)
        else:
            a = cdist(Uniquesample[i].reshape(1,-1), BOX_miu, metric=distancetype)
        
        b = np.sqrt(cdist(Uniquesample[i].reshape(1,-1), BOX_miu, metric='cosine'))
        distance = np.array([a[0],b[0]]).T
        SQ = []
        for j,d in enumerate(distance):
            if d[0] < interval1 and d[1] < interval2:
                SQ.append(j)
        #SQ = np.argwhere(distance[::,0]<interval1 and (distance[::,1]<interval2))
        COUNT = len(SQ)
        if COUNT == 0:
            BOX.append(Uniquesample[i])
            NB = NB + 1
            BOX_S.append(1)
            BOX_miu.append(Uniquesample[i])
            BOX_X.append(sum(Uniquesample[i]**2))
            BOXMT.append(MMtypicality[i])
        if COUNT >= 1:
            DIS = distance[SQ[::],0]/interval1 + distance[SQ[::],1]/interval2
            b = np.argmin(DIS)
            BOX_S[SQ[b]] = BOX_S[SQ[b]] + 1
            BOX_miu[SQ[b]] = (BOX_S[SQ[b]]-1)/BOX_S[SQ[b]]*BOX_miu[SQ[b]] + Uniquesample[i]/BOX_S[SQ[b]]
            BOX_X[SQ[b]] = (BOX_S[SQ[b]]-1)/BOX_S[SQ[b]]*BOX_X[SQ[b]] + sum(Uniquesample[i]**2)/BOX_S[SQ[b]]
            BOXMT[SQ[b]] = BOXMT[SQ[b]] + MMtypicality[i]


    return BOX, BOX_miu, BOX_X, BOX_S, BOXMT, NB


def ChessBoard_PeakIdentification(BOX_miu,BOXMT,NB,Internval1,Internval2, distancetype):
    Centers = []
    n = 2
    ModeNumber = 0
           
    if distancetype == 'minkowski':
        distance1 = squareform(pdist(BOX_miu,metric=distancetype, p=1.5))
    else:
        distance1 = squareform(pdist(BOX_miu,metric=distancetype))        

    distance2 = np.sqrt(squareform(pdist(BOX_miu,metric='cosine')))
      
    for i in range(NB):
        seq = []
        for j,(d1,d2) in enumerate(zip(distance1[i],distance2[i])):
            if d1 < n*Internval1 and d2 < n*Internval2:
                seq.append(j)
        Chessblocak_typicality = [BOXMT[j] for j in seq]

        if max(Chessblocak_typicality) == BOXMT[i]:
            Centers.append(BOX_miu[i])
            ModeNumber = ModeNumber + 1

    return Centers, ModeNumber


def cloud_member_recruitment(ModelNumber,Center_samples,Uniquesample,grid_trad,grid_angl, distancetype):
    L, W = Uniquesample.shape
    Membership = np.zeros((L,ModelNumber))
    Members = np.zeros((L,ModelNumber*W))
    Count = []
    
    if distancetype == 'minkowski':
        distance1 = cdist(Uniquesample,Center_samples, metric=distancetype, p=1.5)/grid_trad
    else:
        distance1 = cdist(Uniquesample,Center_samples, metric=distancetype)/grid_trad

    distance2 = np.sqrt(cdist(Uniquesample, Center_samples, metric='cosine'))/grid_angl
    distance3 = distance1 + distance2
    B = distance3.argmin(1);

    for i in range(ModelNumber):
        seq = []
        for j,b in enumerate(B):
            if b == i:
                seq.append(j)
        Count.append(len(seq))
        Membership[:Count[i]:,i] = seq
        'formato list comprehension so funciona para numpy < 1.18.5'
        #for j in seq:
            #Members[:Count[i]:,W*i:W*(i+1)] = Uniquesample[j]
        Members[:Count[i]:,W*i:W*(i+1)] = [Uniquesample[j] for j in seq]
    MemberNumber = Count
    
    #Converte a matriz para vetor E SOMA +1 PARA NAO TER CONJUNTO 0'
    B = B.A1
    B = [x+1 for x in B]
    return Members,MemberNumber,Membership,B 

def plotar_soda(data,output):
    
    soda_idx = output['IDX'] #lista com as associacoes de conjuntos de cada amostra da serie
    centros = output['C']

    
    T = np.unique(soda_idx)
    
    #OBS: Para dados de teste do SODA inverter data 1 e data2'   
    data1 = data[data.columns[0]].to_numpy()
    data2 = data[data.columns[1]].to_numpy()
    
    soda_idx = list(soda_idx)
    T = list(np.unique(soda_idx))
    
    #Encontra listas de 0 ou 1 para saber em qual conjunto cada valor esta
    result = []
    for a in T:
        sublist = []
        for b in soda_idx:
            if b == a:
                sublist.append(1)
            else: 
                sublist.append(0)
        result.append(sublist)
       
      
    #plt.figure(figsize=(10,5))
    
    #Para usar mais cores na representacao
    import matplotlib.colors as mcolors    
    cores_basicas = mcolors.BASE_COLORS
    colors1 = list(cores_basicas.values())
    cores_tableau = mcolors.CSS4_COLORS
    colors2 = list(cores_tableau.values())    
    colors = colors1[:-2] + colors2[10:60]

    
    #Plota os graficos um de cada vez
    for i in range(len(result)):
    
        auxiliar = [data2[x]*result[i][x] for x in range(len(data2)) ]
        
        for x in range(len(auxiliar)):
            if auxiliar[x] == 0:
                auxiliar[x] = None      #preenche com none para nao poluir o grafico
        plt.plot(data1,auxiliar)
        'ou'
        #plt.scatter(data1,auxiliar)
          
    
    #Metodo alternativo de plotagem USANDO DATAFRAME( nao terminei)'    
    #plt.figure()
    #for t in T:    
        #plt.plot(data[soda_idx == t].n, data[soda_idx == t].avg)
    

    'Para platar os centros dos conjuntos'
    for m in range(len(centros)):
        plt.plot(centros[m][0],centros[m][1],color='black',markersize=8,marker='d')
    

def SelfOrganisedDirectionAwareDataPartitioning(Input, Mode):
    
    """
    Self-organising Direction-Aware Data Partitioning (offline version)
    :params:
    
    :Input: dict containing gridsize, data and distance methodology
    :Mode: Offline or Evolving (online)
    """
    
    
    if Mode == 'Offline':
        data = Input['StaticData']

        L, W = data.shape
        N = Input['GridSize']
        distancetype = Input['DistanceType']
        X1, AvD1, AvD2, grid_trad, grid_angl = grid_set(data,N)
        GD, Uniquesample, Frequency = Globaldensity_Calculator(data, distancetype)

        BOX,BOX_miu,BOX_X,BOX_S,BOXMT,NB = chessboard_division(Uniquesample,GD,grid_trad,grid_angl, distancetype)
        Center,ModeNumber = ChessBoard_PeakIdentification(BOX_miu,BOXMT,NB,grid_trad,grid_angl, distancetype)
        Members,Membernumber,Membership,IDX = cloud_member_recruitment(ModeNumber,Center,data,grid_trad,grid_angl, distancetype)
        
        Boxparameter = {'BOX': BOX,
                'BOX_miu': BOX_miu,
                'BOX_S': BOX_S,
                'NB': NB,
                'XM': X1,
                'L': L,
                'AvM': AvD1,
                'AvA': AvD2,
                'GridSize': N}
        
    if Mode == 'Evolving':
        print(Mode)

    Output = {'C': Center,
              'IDX': IDX,
              'SystemParams': Boxparameter,
              'DistanceType': distancetype}
           
    return Output


def SODA_function(dados, gridsize):
    
    """
    Function to insert information into actual SODA function
    
    :params:
    :data: Can be either: a) array containing the time series historical values
                          b) two-column dataframe
    :gridsize: Decides the level of granularity of the partitioning results. 
    The larger the gridsize is, the more detailed partitioning result the 
    algorithm will obtain.
    """

    #If it is an array or list, make it a two-column dataframe       
    if isinstance(dados, (np.ndarray,list)):
        dados = pd.DataFrame(dados, columns = ['avg'])
        dados.insert(0, '#', range(1,len(dados)+1))

    #Inputs must be a two-collum matrix type variable
    data = np.matrix(dados)

    distances = ['euclidean'];
                

    for d in distances:
        
        Input = {'GridSize':gridsize, 'StaticData':data, 'DistanceType': d}
        
        out = SelfOrganisedDirectionAwareDataPartitioning(Input,'Offline')
        
        saida_idx = out['IDX']

        """
        #Salvando em excel
        nome_arquivo = 'SODA_gridsize_' + str("%d" % gridsize) + '.xlsx'
        output = pd.DataFrame(out['IDX'])
        writer = pd.ExcelWriter(nome_arquivo, engine='xlsxwriter')
        output.to_excel(writer, sheet_name='Geral')
        writer.save()
        """
        
    'nao esta funcionando o plot para a serie diferenciada'
    #plotar_soda(dados,out)
    
    return saida_idx









