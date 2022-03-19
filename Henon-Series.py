# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:21:11 2022

@author: mirla
"""

# Todas as bibliotecas
import numpy as np
import matplotlib.pyplot as plt

#%%
# Função reconst v2 - espaço de estados
def esp_est (serie,dimensao,atraso,ordem=1,hold_out=50):
    N = np.shape(serie)[0] 
    
    P = np.zeros([N-1-(dimensao-1)*atraso,dimensao]) # np.zeros([N-(dimensao-1)*atraso,dimensao])
    linha = 0
    for c in range(0,dimensao):
        P[:,c] = serie[linha:(np.shape(P)[0]+linha)]
        linha = 0 + (atraso*(c+1))
    # Valores desejados - Passo adiante
    T = np.zeros([P.shape[0]])
    for linha in range(0, P.shape[0]):
        T[linha]=serie[(linha+1)+(dimensao-1)*atraso]
    #Treino
    Ptreino=P[:-hold_out,:]
    Ttreino=T[:-hold_out]
    # Teste
    Pteste=P[-hold_out:,:]
    Tteste=T[-hold_out:]
    
    return P, T, Ptreino, Ttreino, Pteste, Tteste

# Função de ativação
def tanh(x,a=1,b=1):
    return np.sinh(x)/np.cosh(x)
    #return np.multiply(a,(1-np.exp(np.multiply(-2*b,x)))/(1+np.exp(-2*b*x))) # Tang. Hiperbólica
def deriv_tanh(x,a=1,b=1):
    return np.multiply(np.multiply((b/a),(a + tanh(x,a,b))),(a - tanh(x,a,b))) # Derivada da Tang. Hiperbólica

#%%

data = np.loadtxt('...henon_200_dados.dat', unpack = True)
x = np.arange(1, len(data)+1, 1)
plt.plot(x,data)
#%%
# Normalização
dados = 2*((data - min(data))/(max(data) - min(data))) -1
plt.plot(x,dados)
plt.show()
#%%
dim = 3
atraso = 1

P, T, Ptreino, Ttreino, Pteste, Tteste = esp_est(dados,dim,atraso,1,50)

# Plot Treino e Teste
print(len(Ptreino),len(Pteste))
x1 = np.arange(0, Ptreino.shape[0], 1)
plt.plot(x1,Ptreino[:,0])
x1 = np.arange(Ptreino.shape[0], P.shape[0], 1)
plt.plot(x1,Pteste[:,0])
plt.show()
#%%
np.random.seed(42)
N_treino = Ptreino.shape[0] 
N_teste = Pteste.shape[0]
variaveis = Ptreino.shape[1] 
# Pesos
bias = 1
neuronios = 5
W = (2*np.random.rand(neuronios,variaveis+1)-1)  # 'neuronios' neuronios na camada oculta, 'variaveis+1' dados (bias + entradas)
V = (2*np.random.rand(1,neuronios+1)-1)  # 1 neuronios de saída, 'neuronios+1' dados (bias + entradas da camada oculta)
# Saidas
Y = np.zeros([1,N_treino]) 
Y2 = np.zeros([N_teste,1]) 
# Número de iterações
epoca_max = 300 
# Taxa de aprendizagem
lr = 0.3 
# Erros
e = np.zeros([1,N_treino])
#%%
for epoca in range(0,epoca_max):
    for i in range(0,N_treino):
        #caminho direto
        #ajuste de coluna de entrada e adiçõ de bias
        entradas = np.append(np.array([bias]),np.transpose(Ptreino[i,:]))
        # combinação linear pesos x entrada = camada oculta
        l1 = np.matrix(np.dot(W,entradas)).T
        # Função de atição
        yl = tanh(l1)
        # Camada oculta - adição de bias
        hl = np.append(np.array([bias]),yl)
        # Combinação com a camada de saída
        U = np.dot(V,hl)
        #saída
        Y[0,i] = tanh(U)
        
        ## Caminho inverso
        # Ajustar os valores de delta dos pesos para a camada de saída
        # Calcula-se o erro
        e[0,i] = Ttreino[i] - Y[0,i]
        # Delta da camada de saída
        delta_Y = e[0,i]*deriv_tanh(U) 
        # Propagar o delta no sentido inverso para a camada oculta
        # Delta da camada oculta
        x1 = np.multiply(delta_Y[0],V[np.newaxis, 0, 1:])
        x2 = np.transpose(deriv_tanh(l1))
        delta_yl = np.matrix(np.multiply(x1,x2))
        # Atualização dos pesos
        V = (V +lr*np.transpose(hl)*delta_Y)
        
        entradas = np.zeros([entradas.shape[0],delta_yl.shape[1]])
        entradas = entradas + entradas.reshape(-1,1)
        
        W = np.add(W ,np.multiply(lr,np.multiply(entradas.T,np.matrix(delta_yl).T)))
#%%
for i in range(0,N_teste):
    #caminho direto
    entradas = np.append(np.array([bias]),np.transpose(Pteste[i,:]))
    # combinação linear pesos x entrada = camada oculta
    l1 = np.matrix(np.dot(W,entradas)).T
    # Função de atição
    yl = tanh(l1)
    # Camada oculta - adição de bias
    hl = np.append(np.array([bias]),yl)
    # Combinação com a camada de saída
    U = np.dot(V,hl)
    #saída
    Y2[i] = tanh(U)
    
erro = np.subtract(Tteste[:], Y2[:,0])
Erro_de_predicao = np.sum(np.power((np.subtract(Tteste[:], Y2[:,0])),2))/np.sum(np.power((np.subtract(Tteste[:], np.mean(Tteste))),2))
    
#%%
xt = np.arange(0, Pteste.shape[0], 1)
plt.plot(xt, Tteste, color='r', linestyle='dashed', label = 'Valor real')
plt.plot(xt, Y2, color='b', label = 'Valor predito')
plt.title('Curva de previsões')
plt.xlabel('Tempo')
plt.ylabel('Valores Normalizados')
plt.legend()
plt.show()

