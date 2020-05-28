from scipy.special import softmax
import numpy as np 
import matplotlib.pyplot as plt

"""
Notação:
x -> input
L -> label
y -> rede output 
A informação do numero de nuronios está na matriz dos pesos, de forma que o número de linhas da matriz W representa o umero de 
nuronios da camada e o numero de colunas dessa matrix representa o numero de neuronios da cama anterior. No caso da primeira
camada o numero de colunas na matriz W representa a quantidade de dados no vertor de input

oraganização e fluxo da rede:

repete até que um parametro arbitrario respeite uma condição. z_1,z_2,...,z_n definem o número de camadas da rede. 
    z_1 = activation( (w_1 * X) + b_1 )
    z_2 = actvation( (w_1 * Z_1) + b_2 )
                .  
                .
                .
    z_n = actvation( (w_n * Z_n-1) + b_n )
    y = actvation( (w_y * Z_n) + b_y )
    loss = diferença( y, L )
    atualização dos pessos (conjunto das matrizes W)
"""

class CreateModel:

    def __init__(self):

        self.bias           = []
        self.layers         = []
        self.activations    = []
        self.df_activations = []
        self.w_step           = 0.1
        self.b_step           = 0.1

    

    ### metodo explicito
    def add_layer(self,N_OF_NEURONS,SHAPE=False):
        """
        este metodo serve para adicionar uma nova camada na rede. N_OF_NEURONS  é o numero de nuronios
        só é necessario adicionar o parametro SHAPE quando for a primeira camada. SHAPE deve informar o formato do input
        """

        if SHAPE:
            self.layers.append(np.random.uniform( -1.0,1.0, (SHAPE[1], N_OF_NEURONS) ).astype('float32'))
            self.bias.append(np.random.uniform(-1.0,1.0, (N_OF_NEURONS,) ).astype('float32'))

        else:
            self.layers.append(np.random.uniform(-1.0,1.0, (self.layers[-1].shape[1], N_OF_NEURONS) ).astype('float32'))
            self.bias.append(np.random.uniform(-1.0,1.0, (N_OF_NEURONS,) ).astype('float32'))
        
    
    ### metodo explicito
    def LoadNet(self, NET):
        """ NET deve ser da forma [LAYERS, BIAS, ACTIVATIONS] """

        if len(NET) == 3:
            self.layers = NET[0]
            self.bias = NET[1]
            self.activations = NET[2]
        else:
            raise Exception("NET size doesnt mach")

    ### metodo explicito
    def add_activation(self,activation = "relu"):
        """Este metodo serve para adicionarmos um tipo de ativação de cada camada suas respectivas derivadas"""

        if activation is "relu":
            self.activations.append(self.relu)
            self.df_activations.append(self.df_relu)
        elif activation is "sigmoid":
            self.activations.append(self.sigmoid)
            self.df_activations.append(self.df_sigmoid)
        elif activation is "iden":
            self.activations.append(self.iden)
            self.df_activations.append(self.df_iden)
        elif activation is "softmax":
            self.activations.append(self.Softmax)
            self.df_activations.append(self.df_Softmax)        
        else:
            raise Exception('Non-supported activation function')

    # estas são as duas funções de ativação   
    ### metodo implicito 
    def Softmax(self,x):
        return softmax(x,axis=1)

    def df_Softmax(self, x):
        return softmax(x)*(1.0-softmax(x,axis=1))

    def relu(self,x):
        return x*(x>0)

    def df_relu(self, x):
        return 1*(x>0)
    
    def sigmoid(self,x):
        return 1./(1.+np.exp(-x))

    def df_sigmoid(self, x):
        return np.exp(-x)/((1+np.exp(-x))**2)
    
    def iden(self,x):
        return x
    
    def cost(self,d,t):
        return ((d-t)**2).sum() / len(d)

    def Predict(self, data):

        self.propagations = []

        #calcula z da primeira camda e as derivadas com relação a z
        Z = np.dot(data, self.layers[0]) + self.bias[0]
        
        z_atual =  self.activations[0](Z)
        self.propagations.append(z_atual)
        #print(z_atual)

        for i in range( 1, len(self.layers)):

            Z = np.dot(z_atual, self.layers[i]) + self.bias[i]
            z_atual = self.activations[i](Z)
            self.propagations.append(z_atual)

        return z_atual


    ### metodo implicito
    def forward_propagation(self, data):

        self.derivatives    = []
        self.propagations   = []

        #calcula z da primeira camda e as derivadas com relação a z
        Z = np.dot(data, self.layers[0]) + self.bias[0]
        self.derivatives.append(self.df_activations[0](Z))
        
        z_atual =  self.activations[0](Z)
        self.propagations.append(z_atual)
        #print(z_atual)

        for i in range( 1, len(self.layers)):

            Z = np.dot(z_atual, self.layers[i]) + self.bias[i]
            self.derivatives.append(self.df_activations[i](Z))
            z_proximo = self.activations[i](Z)
            z_atual = z_proximo
            self.propagations.append(z_atual)


    ### metodo explicito
    def back_propagation(self,data, target, batch_size):

        self.dc_dw          = []
        self.dc_db          = []
        

        # multiplicação entrada por entrada
        delta = ( self.propagations[-1] - target)*self.derivatives[-1]
        self.dc_dw.append( np.dot(np.transpose(self.propagations[-2]), delta )/batch_size )
        self.dc_db.append( delta.sum(0)/batch_size )
        
        # vai do penultimo layer até o primeiro, o primeiro layer pe o layer apos o input!
        # [n-1,n-2,...,0]
        for i in np.arange(len(self.layers)-1, 1,-1):
            # 2

            delta = np.dot(delta, np.transpose(self.layers[i]))*self.derivatives[i-1]
            self.dc_dw.append( np.dot(np.transpose(self.propagations[i-2]), delta )/batch_size )
            self.dc_db.append( delta.sum(0)/batch_size )

        delta = np.dot(delta, np.transpose(self.layers[1]))*self.derivatives[0]
        self.dc_dw.append( np.dot(np.transpose(data),delta)/batch_size )
        self.dc_db.append( delta.sum(0)/batch_size )

        ### atualiza os pesos
        for i in range(len(self.layers)):

            self.layers[i] -= self.w_step*self.dc_dw[-i-1]
            self.bias[i] -=  self.b_step*self.dc_db[-i-1]


        return self.cost( self.Predict(data), target )

    def fit(self, x, target, epochs, batch_size):

        #x and targe are the full datasets
        self.x = x
        self.target = target
        
        d_size = len(self.x)

        if d_size%batch_size or len(self.target)%batch_size:
            raise Exception(
                'Data and Target size must be a multiple of batch_size (data_size %% batch_size == 0 ) ')
        elif len(self.x) != len(self.target):
            raise Exception(
                'Data and target must be the same size')
        else:

            batches = d_size//batch_size

            for i in range(epochs):

                # shuffle data every epoch
                indx = np.random.permutation(d_size)
                self.x, self.target = self.x[indx], self.target[indx]

                #split data and target in epochs with batch size iquals to size/batch
                target_splited = np.split(self.target, batches)
                data_splited = np.split(self.x, batches)
                

                for j in range(batches):
                                
                    self.forward_propagation(data_splited[j])
                    cost = self.back_propagation(data_splited[j], target_splited[j], batch_size)
                    
                
                    if cost < 1.0e-4:
                        self.SaveNet(f'2dnet/2dNET_epoch{i}_{j}')
                        print(f'Saving in Empoch{i}_{j}, c={cost}')
                

                if ((i+1)%100 == 0):
                    #self.SaveNet(f'2dnet/2dNET_epoch{i}')
                    #print(f'Saving in empoch{i}, c={cost}')
                    print(f'Epoch: {i+1} of {epochs}')
                    #print( '['+'#'*(i+1)+' '*(epochs-1-i)+']' )
                    print(f'loss: { cost }\n')

    def SaveNet(self, NAME):

        model_data = [self.layers, self.bias, self.activations]
        np.save(f'{NAME}.npy', model_data)

