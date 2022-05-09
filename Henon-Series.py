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
def separa(passo, proxpasso, separacao=50):
    #Treino
    passo_treino=passo[:-separacao,:]
    proxpasso_treino=proxpasso[:-separacao]
    # Teste
    passo_teste=passo[-separacao:,:]
    proxpasso_teste=proxpasso[-separacao:]
    
    return passo_treino, proxpasso_treino, passo_teste, proxpasso_teste

def esp_est (serie,dimensao,atraso,separacao=50):
    N = np.shape(serie)[0] 
    
    passo = np.zeros([N-1-(dimensao-1)*atraso,dimensao]) # np.zeros([N-(dimensao-1)*atraso,dimensao])
    linha = 0
    for c in range(0,dimensao):
        passo[:,c] = serie[linha:(np.shape(passo)[0]+linha)]
        linha = 0 + (atraso*(c+1))
    # Valores desejados - Passo adiante
    proxpasso = np.zeros([passo.shape[0]])
    for linha in range(0, passo.shape[0]):
        proxpasso[linha]=serie[(linha+1)+(dimensao-1)*atraso]
    
    passo_treino, proxpasso_treino, passo_teste, proxpasso_teste = separa(passo, proxpasso, separacao=50)
    
    return passo, proxpasso, passo_treino, proxpasso_treino, passo_teste, proxpasso_teste

# Função de ativação
def tanh(x,a=1,b=1):
    return np.sinh(x)/np.cosh(x)
    #return np.multiply(a,(1-np.exp(np.multiply(-2*b,x)))/(1+np.exp(-2*b*x))) # Tang. Hiperbólica
def deriv_tanh(x,a=1,b=1):
    return np.multiply(np.multiply((b/a),(a + tanh(x,a,b))),(a - tanh(x,a,b))) # Derivada da Tang. Hiperbólica

#%%
data = np.loadtxt('henon_200_dados.dat', unpack = True)
x = np.arange(1, len(data)+1, 1)
fig = plt.figure(figsize=(6, 3))
plt.plot(x,data)
plt.title('Dados',fontsize = 18)
plt.show()
#%%
# Normalização
dados = 2*((data - min(data))/(max(data) - min(data))) -1 # -1 a 1
#%%
dim = 4
atraso = 1

passo, proxpasso, passo_treino, proxpasso_treino, passo_teste, proxpasso_teste = esp_est(dados,dim,atraso,50)

# Plot Treino e Teste
#print(len(passo_treino),len(passo_teste))
xx = np.arange(0, passo_treino.shape[0], 1)
fig = plt.figure(figsize=(6, 3))
plt.plot(xx,passo_treino[:,0],label='treino')
xx = np.arange(passo_treino.shape[0], passo.shape[0], 1)
plt.plot(xx,passo_teste[:,0],label='teste')
plt.title('Dados treino e teste',fontsize = 18)
plt.legend()
plt.show()
#%%
n_treino = passo_treino.shape[0] 
n_teste = passo_teste.shape[0]
n_classes = passo_treino.shape[1] 

# Pesos
bias = 1
neuronios = 4
np.random.seed(42)
w = (2*np.random.rand(neuronios,n_classes+1)-1)  # 'neuronios' neuronios na camada oculta, 'n_classes+1' dados (bias + ent)
v = (2*np.random.rand(1,neuronios+1)-1)  # 1 neuronios de saída, 'neuronios+1' dados (bias + entradas da camada oculta)

# Saidas
y = np.zeros([1,n_treino]) 
y_teste = np.zeros([n_teste,1]) 

# Número de iterações
epoca_max = 400 

# Taxa de aprendizagem
taxaAprendizagem = 0.2 

# Erros
e = np.zeros([1,n_treino]) #armazenara os erros em cada época
eqm = np.zeros([1,epoca_max]) # armazenará os erros quadráticos médios
#%%
for epoca in range(0,epoca_max): #treinamento
    for i in range(0,n_treino):
        #CAMINHO DIRETO
        #ajuste de coluna de entrada e adição de bias
        ent = np.append(np.array([bias]),np.transpose(passo_treino[i,:]))
        #combinação linear de pesos e entrada 
        soma1 = np.matrix(np.dot(w,ent)).T
        y1 = tanh(soma1) # saida da 1° camada com a f.ativação
        # Na Camada oculta -> adição de bias
        camadaOculta = np.append(np.array([bias]),y1)
        # Combinação linear
        soma2 = np.dot(v,camadaOculta)
        #saída
        y[0,i] = tanh(soma2)
        
        # Calcula-se o erro
        e[0,i] = proxpasso_treino[i] - y[0,i]
        
        #CAMINHO INVERSO
        # A partir do erro, definir o delta da saída
        delta_y = e[0,i]*deriv_tanh(soma2) 
        # delta_y será propagado no sentido inverso a camada oculta
        delta_peso_saida = np.multiply(delta_y[0],v[np.newaxis, 0, 1:])
        delta_oculta = np.transpose(deriv_tanh(soma1))
        delta_y1 = np.matrix(np.multiply(delta_peso_saida,delta_oculta))
        
        # Atualização dos pesos
        v = (v + np.multiply(taxaAprendizagem, np.multiply(np.transpose(camadaOculta),delta_y)))
        w = np.add(w ,np.multiply(taxaAprendizagem,np.multiply(ent.T,np.matrix(delta_y1).T)))
    
    eqm[0,epoca] = np.multiply(1/(n_treino),(np.sum(np.power(e,2))))
#%%
#plotagem do eqm
fig = plt.figure(figsize=(8, 4))
plt.plot(np.arange(epoca_max),eqm[0])
plt.title('Erro ao longo das épocas',fontsize = 18)
plt.xlabel('Época',fontsize = 16)
plt.ylabel('Erro quadrático médio',fontsize = 16)
plt.show()

#%%
for i in range(0,n_teste):
    #CAMINHO DIRETO
    #ajuste de coluna de entrada e adição de bias
    ent = np.append(np.array([bias]),np.transpose(passo_teste[i,:]))
    #combinação linear de pesos e entrada 
    soma1 = np.matrix(np.dot(w,ent)).T
    y1 = tanh(soma1) # saida da 1° camada com a f.ativação
    # Na Camada oculta -> adição de bias
    camadaOculta = np.append(np.array([bias]),y1)
    # Combinação linear
    soma2 = np.dot(v,camadaOculta)
    #saída
    y_teste[i] = tanh(soma2)
    
erro = np.subtract(proxpasso_teste[:], y_teste[:,0])
erro_teste = np.sum(np.power((np.subtract(proxpasso_teste[:], y_teste[:,0])),2))/np.sum(np.power((np.subtract(proxpasso_teste[:], np.mean(proxpasso_teste))),2))
print('Erro da predição do teste: ', erro_teste) 
#%%
#Plotagem dos valores previsos no teste
fig = plt.figure(figsize=(8,5))
xt = np.arange(0, passo_teste.shape[0], 1)
plt.plot(xt, proxpasso_teste, color='r', linestyle='dashed', label = 'Valor real')
plt.plot(xt, y_teste, color='b', label = 'Valor predito')
plt.title('Curva de previsões',fontsize = 18)
plt.xlabel('Tempo',fontsize = 16)
plt.ylabel('Valores Normalizados',fontsize = 16)
plt.legend(loc='lower left')
plt.show()
#%%

