
import math
import random
import time
import numpy

def main():

    bolean = True
    while bolean == True:
        print()
        print ('EP1 - MAP3121 - Métodos Numéricos e Aplicações')
        escolhido = int(input('Para Testes, digite 1, para Classificar, 2, para Sair, 3: '))
        print()

        if escolhido == 1:
            print ('1- Teste A')
            print ('2- Teste B')
            print ('3- Teste C')
            print ('4- Teste D')
            print ('5- Teste segunda tarefa')
            y = int(input('R:'))
            if y == 1:
                print(Resolve_Sistema(testeA(64,64)[0],testeA(64,64)[1],64,64))
            if y == 2:
                print(Resolve_Sistema(testeB(20,17)[0],testeB(20,17)[1],20,17))
            if y == 3:
                print(SistSimultaneos(testeC(64,64,3)[0],testeC(64,64,3)[1]))
            if y == 4:
                print(SistSimultaneos(testeD(20,17,3)[0],testeD(20,17,3)[1]))
            if y == 5:
                C = [[3/10,3/5,0],[1/2,0,1],[4/10,4/5,0]]
                W,H,d = FATORAÇÃO(C,3,3,2)
                print ('W =',W)
                print ('H =',H)            
        if escolhido == 2:
            p = int(input('p:'))
            n_test = int(input('n_test:'))
            ndig_treino = int(input('ndig_treino:'))
            print()

            Teste(ndig_treino,n_test,p)
            print()
        if escolhido == 3:
              bolean = False
        else:
              None

 #_____________________________________________________________________________#   


def Teste(ndig_treino,n_test,p):
    '''
    ndig_treino é com quantas imagens o dig sera testado/comparado
    n_test é quantas imagens serao comparadas
    p é quantas componentes terao Wd
    '''
    
    A = numpy.loadtxt('test_images.txt',usecols = range(n_test))
    Wd = TreinaDig(ndig_treino,p)[1] #TreinaDig me devolve uma linha com 
    # treinei todos os digitos        # as Wd(matriz) de cada digito em cada posição
    Hd = []
    for g in range(10): 
        Hdig = SistSimultaneos(Wd[g],A)
        Hd.insert(g,Hdig)
    
    WdigHdig =[]
    for nossa in range(10):
        WdigHdig.insert(nossa,Wd[nossa]@Hd[nossa])
        #agora tenho uma lista com matrizes resultado da multiplicação de 
        # matrizes W e H para cada digito, WdigHdig é 784x784
    
    normas = [] #tera 2 dimensoes, digitos e erros das imagens(colunas)
    # 10xn_test
        # essa lista terá em suas linhas as informações de cada digito
        # dentro de cada linha tera as normas de cada imagem qnd comparadas
        # com cada digito, é uma matriz importante    
    for dig in range(10):   # primeiro, escolhe qual digito vai ser comparado
        WHdig = WdigHdig[dig] #para dig 0, dps 1,2,3...
        
        for a_colunas in range(len(A[0])): # len(A[0]) == n_test 
            normI = []       #pra uma imagem
            for a_linhas in range(len(A)-1):
                # esse laço coloca varias diferenças entre a imagem(coluna)
                # e a aproximaçao (WHdig) na lista normI
                normI.append(abs(A[a_linhas][a_colunas]-WHdig[a_linhas][a_colunas]))
                # normI é uma linha, calculamos a norma mesmo assim
            c = norma_euclidiana(normI)
                #uma imagem de 784 linhas é reduzida a um valor
                #AQUI CABE UMA ANALISE:
#  o valor c (norma euclidiana de normI) é a dif quadratica, um erro, entre 
#   A IMAGEM REAL, E A APROXIMAÇÃO WdHd, ou seja, é o parametro
#   de comparação
            normI.append(c) #normI é quanto cada imagem se parece com o digito
        
        
        normas.append(normI)

    classific, Eassoc = analisaErro(normas,n_test)
    #sera feita a analise de todos os erros devolvendo duas matrizes, classific
    # com a classificação de cada imagem como um tal digito e Eassoc com o erro
    # de classificação associado a cada uma delas
    
    reais, corretas  = validacao(classific,Eassoc)
    
    for i in range (len(reais)):
        if reais[i] != 0:
            percent = corretas[i]*100/reais[i]
            print ('o digito %d teve %d acertos, representando %f' %(i, corretas[i], percent))
        else:
            percent = 0
            print ('o digito %d teve %d acertos, representando %f' %(i, corretas[i], percent))
        
    return
    
 #_____________________________________________________________________________#   

def validacao(classific,Eassoc):
    '''
    classific é n_test X 1
    Eassoc é n_test X 1
    '''
    
    Real = numpy.loadtxt('test_index.txt')
    # eh uma vetor lista
    
    reais = []
    corretas = []
    
    for dig in range(10): 
        deveria = 0
        acertos = 0
        for d in range (len(classific)): #==n_test
            if classific[d] == Real[d] == dig:
                acertos += 1
            elif Real[d] == dig:
                deveria +=1

        reais.append(deveria)
        corretas.append(acertos)
    
    return reais, corretas                       
 #_____________________________________________________________________________#   

def analisaErro(normas,n_test):
    '''
    inicialmente A imagens é consideradas 0
    depois da verificação dos erros com os outros digitos, podem se tornar 
    outros digitos
    '''
    
    classific = []
    Eassoc = []

    for im in range(len(normas[0])):  #pra uma imagem...  #normas[0]==n_test
        menor = 1000
        guard = 0
        for d in range (len(normas)): #procurar entre os digitos qual o menor erro
            if normas[d][im] <= menor: #d so vai de 0 a 9
                menor = normas[d][im]  # encontramos o menor erro da imagem
                guard = d
        # visto todos os erros pra uma imagem
        classific.append(guard)
        Eassoc.append(menor)
        # dois vetores linha com qual digito cada imagem representa
        # e outra com o erro (probabilidade) de ser esse digito
    return classific, Eassoc

 #_____________________________________________________________________________#   
def TreinaDigUSER(dig,ndig_treino,p):
  
    t0 = time.process_time()
    
    Mat_digX = []
    
    print('Treinando digito', dig)       
    
    traindig = numpy.loadtxt('train_dig'+str(dig)+'.txt',usecols = range(ndig_treino))
    
    Wd = FATORAÇÃO(traindig,len(traindig),len(traindig[0]),p)[0]
    
    Mat_digX.insert(dig,Wd)
        
    t1 = time.process_time()
    
    d = t1 - t0
    
    print('tempo para treinar dig',dig,':', d)

    return d, Mat_digX
 #_____________________________________________________________________________#   
    
def TreinaDig(ndig_treino,p):
    '''
    cada imagem é uma (coluna de 784 linhas) e m colunas (n_cols, ou, ndig_treinos)
    dai forma uma matriz com a aprendizagem do digito Wd com (n=784)Xp (p escolhido)
    Guardar cada Wd

    '''
    t0 = time.process_time()
    Mat_digX = []
    for dig in range(10):
        tin = time.process_time()
        print('Treinando digito', dig)       
        traindig = numpy.loadtxt('train_dig'+str(dig)+'.txt',usecols = range(ndig_treino))
        for i in range (len(traindig)):
            for j in range (len(traindig[0])):
                traindig[i][j] = traindig[i][j]/255
        Wd = FATORAÇÃO(traindig,len(traindig),len(traindig[0]),p)[0]
        Mat_digX.insert(dig,Wd)
        tout = time.process_time()
        delta = tout - tin
        print('tempo para treinar dig',dig,':', delta)
    t1 = time.process_time()
    d = t1 - t0
    return d, Mat_digX
    
 #_____________________________________________________________________________#   

def FATORAÇÃO(A,n,m,p):
    '''A quadrado nXm
        W nXp
        H pxm
    '''
    t0 = time.process_time()
    W = matrixAleatoria255(n,p)
    Acopia = copiaMatriz(A)
    e = CalculaErro(A,W)
    it = 0
    while e > 10**-5 and it < 100:
        for col in range(len(W[0])):  # normalizando as Colunas
            s = 0
            for lin in range(len(W)):  # soma valores da coluna
                s += (W[lin][col])**2
            sq = math.sqrt(s)
            W = Normaliza_Coluna(W,col,sq) #normaliza a coluna de W
    
        H = numpy.array(SistSimultaneos(W,A))  # com W normalizado e A
        H = redefine(H)
        Wa = numpy.array(W)
        Ha = numpy.array(H)
        WH = Wa@Ha
        e0 = CalculaErro(A,WH)
        Ht = H.transpose()
        Acopiaa = numpy.array(Acopia)
        At = Acopiaa.transpose()
        Wt = numpy.array(SistSimultaneos(Ht,At))
        W = Wt.transpose()
        W = redefine(W)
        H = Ht.transpose()
        WH = W@H   # Multplica W por H
        e1 = CalculaErro(A,WH)

        e = abs(e0-e1)
               
        it += 1
        
    t1 = time.process_time()
    d = t1 - t0
    return W,H,d


 #_____________________________________________________________________________#   


def SistSimultaneos(W,A):
    '''
    recebe uma matriz A e uma W, e devolve uma H, matriz de solução
    de todas os sistemas W com A[i]
    '''
    
    Wcopia = copiaMatriz(W)
    H = []
    for a_colunas in range(len(A[0])):    #formando uma matriz temporaria
        temp = []                      #com a coluna desejada para solve sist
        for a_linhas in range(len(A)):  #Destacamos uma coluna
            temp.append([A[a_linhas][a_colunas]])
            Wcopia = copiaMatriz(Wcopia)
            
        x = Resolve_Sistema(Wcopia,temp,len(Wcopia),len(Wcopia[0])) # matriz coluna
        x = numpy.array(x)
        x = x.transpose()       # agora matriz linha
        H.append(x[0])
    Ha = numpy.array(H)
    return Ha.transpose()

 #_____________________________________________________________________________#   

def Resolve_Sistema(W,b,n,m):
    if n!= m :
        W,b = Sobredeterminada(W,b)
    m = len(W[0])
    n = len(W)
    for k in range(len(W[0])):
        for j in range(n-1,k,-1):
            i = j-1
            while W[j][k] > 10**-50:        #so chama rot_givens qnd tem que fazer operação
                if abs(W[i][k])>abs(W[i+1][k]): #caso a entrada for 0, continua procurando
                    tau = -W[i+1][k]/W[i][k]
                    c = 1/math.sqrt(1+tau**2)
                    s = c*tau
                    Rot_givens(W,n,m,i,j,c,s)
                    Rot_givens(b,n,0,i,j,c,s)
                else:
                    tau = -W[i][k]/W[i+1][k]
                    s = 1/math.sqrt(1+tau**2)
                    c = s*tau                              
                    Rot_givens(W,n,m,i,j,c,s)
                    Rot_givens(b,n,0,i,j,c,s)
    
    #agora vamos resolver o sistema com a matriz escalonada
                    
    q = []  
    if W[n-1][m-1] != 0: #ultimo valor da diagonal principal
        sol0 = (b[n-1][0])/W[n-1][m-1]     #CASO PARTICULAR DA 
    else:                               # ULTIMA LINHA
        sol0 = 0
    q.append([sol0])  # uma linha eh adc ao vetor de soluções
                    #na ultima posição
    
    for linha in range(n-1,0,-1):   #monta a matriz x das soluçoes
        soma = 0        # loop de baixo a cima, resolvendo o sistema
        u = 0
        for j in range (linha,m):
            soma += W[linha-1][j]*q[u][0]
            u += 1

        if W[linha-1][linha-1] != 0: #caso o elemento da diag prin for 0
            soli = (b[linha-1][0]-soma)/W[linha-1][linha-1]
        else:           #caso geral
            soli = 0
        q.insert(0,[soli]) #adc na posição inicial as soluções das linhas
             
    return q    #devolve uma matriz coluna de soluçao

 #_____________________________________________________________________________#   

def Rot_givens(W,n,m,i,j,c,s):

    if m == 0:  #RotGivens pra matriz coluna
        aux = c*W[i][0]-s*W[j][0]
        W[j][0] = s*W[i][0]+c*W[j][0]
        W[i][0] = aux
        
    else:   #RotGivens pra matriz a ser escalonada
        for k in range (0,m):
            aux = c*W[i][k]-s*W[j][k]
            W[j][k] = s*W[i][k]+c*W[j][k]
            W[i][k] = aux
                     
    return W

 #_____________________________________________________________________________#   

def Sobredeterminada(W,b):
    Wa = numpy.array(W)
    ba = numpy.array(b)
    Wt = Wa.transpose()
    R = Wt@W
    b = Wt@ba
                #daqui sai a matriz R quadrada
    return R,b #b so é corrigida pela multip. de matrizes

 #_____________________________________________________________________________#   

def Normaliza_Coluna(W,col,sq):
    for lin in range(len(W)):  # normaliza todas as entradas dessa coluna
        if sq == 0:
            W[lin][col] = 0.0
        else:
            W[lin][col] = W[lin][col]/sq

    return W

 #_____________________________________________________________________________#   

def copiaMatriz(W):
    Wcop = []   
    for i in range(len(W)):
        Wcop.append([])
        for j in range(len(W[i])):
            Wcop[i].append(W[i][j])  #adiciona todas as entradas de uma coluna
                                    # numa linha
    return Wcop

 #_____________________________________________________________________________#   

def matrixAleatoria9(n,m):
    E = []
    for a_linhas in range(n): 
        temp = []
        for a_colunas in range(m): 
            x = random.randrange(0,9)
            temp.append(x)
        E.append(temp)

    return E

 #_____________________________________________________________________________#

def matrixAleatoria255(n,m):
    E = []
    for a_linhas in range(n): 
        temp = []
        for a_colunas in range(m): 
            x = random.randrange(0,255)
            temp.append(x)
        E.append(temp)

    return E

 #_____________________________________________________________________________#   

def CalculaErro(A,WH): #A - WH

    E =[]
    for linhas in range(len(WH)):  #nesse passo A e WH tem mesma dimensao
        for colunas in range(len(WH[0])):
            Etemp = (A[linhas][colunas]-WH[linhas][colunas])**2 #variavel
            E.append(Etemp) # coloca as pequenas diferencas na lista
    erro = math.sqrt(sum(E))

    return erro

 #_____________________________________________________________________________#   

def redefine(M):
    for a_linhas in range(len(M)):
        for a_colunas in range(len(M[0])):  
            if M[a_linhas][a_colunas] <= 0:
                M[a_linhas][a_colunas] = 0
    return M

 #_____________________________________________________________________________#   

def norma_euclidiana(A):
    E = numpy.linalg.norm(A)
    return E

 #_____________________________________________________________________________#   

def testeA(n,m):
    W=[]
    for i in range (n):
        linhai= []
        for j in range (m):
            if i == j:
                linhai.append(2)
            elif abs(i-j) == 1:
                linhai.append(1)
            elif abs(i-j) > 1:
                linhai.append(0)
                           
        W.append(linhai)
        
    b = []
    for i in range (0,n):
        b.append([1])

    return W,b

 #_____________________________________________________________________________#   

def testeB(n,m):
    W=[]
    for i in range (0,n):
        linhai= []
        for j in range (0,m):
            if abs(i-j) <=4:
                if i+j-1 == 0:
                    linhai.append(0)
                else:
                    linhai.append(1/(i+j-1))
            elif abs(i-j) >4:
                linhai.append(0)
                           
        W.append(linhai)

    b = []        
    for i in range (0,n):
        b.append([i])
    
    return W,b

 #_____________________________________________________________________________#   

def testeC(n,p,m):
    W=[]
    for i in range (n):
        linhai= []
        for j in range (p):
            if i == j:
                linhai.append(2)
            elif abs(i-j) == 1:
                linhai.append(1)
            elif abs(i-j) > 1:
                linhai.append(0)
                           
        W.append(linhai)
        
    A = []
    for k in range (n):
        linhak= []
        for j in range (m+1):
            if j == 1:
                linhak.append(1)
            elif j == 2:
                linhak.append(k)
            elif j == 3:
                linhak.append(2*k-1)
        A.append(linhak)

    return W, A

 #_____________________________________________________________________________#   

def testeD(n,p,m):
    W=[]
    for i in range (0,n):
        linhai= []
        for j in range (0,p):
            if abs(i-j) <=4:
                if i+j-1 == 0:
                    linhai.append(0)
                else:
                    linhai.append(1/(i+j-1))
            elif abs(i-j) >4:
                linhai.append(0)
                           
        W.append(linhai)

    A = []        
    for k in range (n):
        linhak= []
        for j in range (m+1):
            if j == 1:
                linhak.append(1)
            elif j == 2:
                linhak.append(k)
            elif j == 3:
                linhak.append(2*k-1)
        A.append(linhak)
        
    return W, A

main()