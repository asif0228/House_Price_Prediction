'''
* This is a Batch Gradient Descent Implementation
* with only 1 feature
* we will have only 1 layer of neural netwok

00. Main Function
01. Load data.csv and remove 1st column which is serial
02. Check for missing data
03. Normalize data, formula df_norm = (df - df.mean())/df.std() Here, std = standard deviation
04. as we normalized result (y) also so need a function to convert back our pridected result (y) to real y
05. Seperate x and y.
06. Seperate Training and Test set
07. Create NN Model
- 07.01 Initilize and set weights
08. Tain The Neural Netwok
- 08.01 Start Training NN for given iteration
- 08.02 Compute Cost
- 08.03 Update Weight Using Batch Gradient Descent
09. Plot cost vs iteration gaph
10. Predict using the trained Model
'''

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt


class NeuralNetworkGD:
    ### 07.01 Initilize and set weights
    def __init__(self, lc, nc, tx, ty, lr):
        self.layerCnt = lc # Layer Count
        self.nodeCnt = nc # Node Count for rach layer
        self.iteration = np.array([]) # Iteration array for plotting
        self.cost = np.array([]) # Cost of each layer to minimize
        self.learningRate = lr # learning rate
        self.finalesult = []
        # Generate random weight for each node of each layer
        self.weights = {}
        for i in range(0,self.layerCnt):
            self.weights["w"+str(i)] = np.random.uniform(0,1,(self.nodeCnt[i]+1,self.nodeCnt[i+1])) # each will have weight for a biased node
        self.train_x = tx # set Input
        self.train_y = ty
    
    # If user wants to re initilize weights
    def reInitilize(self):
        self.weights = {}
        for i in range(0,self.layerCnt):
            self.weights["w"+str(i)] = np.random.uniform(0,1,(self.nodeCnt[i]+1,self.nodeCnt[i+1])) # each will have weight for a biased node
        
    ### 08.01 Start Training NN for given iteration
    def train(self, iteration):
        results = {}
        for j in range(0,iteration):
            # self.reInitilize()
            inp = np.append(self.train_x, np.ones((self.train_x.shape[0],1), dtype="int64"), axis=1) # set Input with bias
            for i in range(0,self.layerCnt):
                self.finalesult = np.dot(inp,self.weights["w"+str(i)]) # calculate result for ith layer
                results["r"+str(i)] = self.rectify(self.finalesult) # calling activation function
                #print(self.finalesult.shape)
                inp = np.append(results["r"+str(i)], np.ones((results["r"+str(i)].shape[0],1), dtype="int64"), axis=1) # set Input with bias
                #print(inp.shape)
            self.costFunction()# update weights
            self.iteration = np.append(self.iteration,[j+1])
            if len(self.cost)>1:
                if self.cost[len(self.cost)-1] >= self.cost[len(self.cost)-2]:
                    print("Terminating NN Training at iteration: "+str(j))
                    #self.finalesult = results["r"+str(self.layerCnt-1)]
                    return ""
            self.updateWeight()# update weights
        #self.plotCostFunction()
        print("Terminating NN Training after full iteration")
        #self.finalesult = results["r"+str(self.layerCnt-1)]
        return ""
    
    ### 08.02 Compute Cost
    def costFunction(self):
        tmpcost = self.finalesult - self.train_y
        tmpcost = np.sum((tmpcost*tmpcost))/(2*tmpcost.shape[0])
        self.cost = np.append(self.cost,[tmpcost])
        return ""
    
    ### 08.03 Update Weight Using Batch Gradient Descent
    def updateWeight(self):
        print("--=-- UPDATE WEIGHT --=-- : "+str(self.iteration[len(self.iteration)-1])+" : "+str(self.cost[len(self.cost)-1]))
        tmpcost = {}
        m = self.finalesult.shape[0]
        tmp = self.finalesult - self.train_y.reshape(self.train_y.shape[0], 1)
        
        for i in range(0,self.weights["w0"].shape[0]):
            if i == self.weights["w0"].shape[0]-1:
                tmpcost[i] = np.sum(tmp) # Biased weight
            else:
                tmpcost[i] = np.sum(tmp * self.train_x) # As there is only one feature
            tmpcost[i] = np.divide(tmpcost[i],m)
            tmpcost[i] = tmpcost[i] * self.learningRate
            self.weights["w0"][i] -= tmpcost[i]
            
        return ""
    
    ### 09. Plot cost vs iteration gaph
    def plotCostFunction(self):
        #return ""
        # plotting the points  
        plt.plot(self.iteration, self.cost) 
          
        # naming the x axis 
        plt.xlabel('x - Iteration') 
        # naming the y axis 
        plt.ylabel('y - Cost') 
          
        # giving a title to my graph 
        plt.title('Cost vs Iteration') 
          
        # function to show the plot 
        plt.show()
        ### =====================================
        # plotting the points
        
        # giving a title to my graph 
        plt.title('oiginal vs Learned Result') 
         
        #plt.plot( self.iteration, label="X", color="ed" )
        plt.plot( self.finalesult, label="Y", color="blue" )
        plt.plot( self.train_y, label="Z", color="green" )
        plt.show()
        
        ### ========================================
        rng = np.array([x for x in range(0,self.train_y.shape[0])])
        fig=plt.figure()
        ax=fig.add_axes([0,0,1,1])
        ax.scatter(rng, self.train_y, color='r')
        ax.scatter(rng, self.finalesult, color='b')
        ax.set_xlabel('0 to 1')
        ax.set_ylabel('Normalized House Price')
        ax.set_title('oiginal vs Learned Result')
        plt.show()
        
        
        return ""
    
    def getFinalResult(self):
        return self.finalesult
    
    def describe(self):
        print("== Describing Neural Network ==")
        print("Layer Count: "+str(self.layerCnt))
        print("Node Count for rach layer")
        print(self.nodeCnt)
        print("Training Set X")
        print(self.train_x[:3])
        print("Output Set Y")
        print(self.train_y[:3])
        print("Weight of all nodes")
        print(self.weights["w0"])
        #for i in range(0,self.layerCnt):
        #    print(self.weights["w"+str(i)].shape)
        print("===========================")
        
    def rectify(self, val):
        return val.clip(min=0)



### 01. Load data.csv and remove 1st column which is serial
def loadData():
    return genfromtxt('data_single_feature.csv', delimiter=',',dtype='int')[:,1:]

### 02. Check for missing data
def checkForMissingData(my_data):
    i=0
    for row in my_data:
        if -1 in row:
            my_data = np.delete(my_data,(i),0)
            i-=1
        i+=1
    return my_data

### 03. Normalize data, formula df_norm = (df - df.mean())/df.std() Here, std = standard deviation
def normaLizeData(my_data):
    return np.divide(np.subtract(my_data,np.mean(my_data,0)),np.std(my_data,0))

### 04. as we normalized result (y) also so need a function to convert back our pridected result (y) to real y
def deNormalizeY(my_data,y_val):
    return (y_val*np.std(my_data))+np.mean(my_data)
    #return int(np.add(np.multiply(y_val,np.std(my_data)),np.mean(my_data)))

### 05. Seperate x and y.
def seperateXY(my_data_norm):
    return my_data_norm[:,:my_data_norm.shape[1]-1],my_data_norm[:,-1]

### 06. Seperate Training and Test set
def seperateTrainAndTestSet(x,y):
    train = x.shape[0]-int(x.shape[0]*0.05) ### 0.05 = 5% test data 95 % train data
    return x[:train,:],x[train:,:],y[:train],y[train:]


### 07. Create NN Model
def createNNModel(lc,nc,tx,ty,lr): # lc = Layer Count, nc = Node Count for each layer, lr = learning rate
    nn = NeuralNetworkGD(lc,nc,tx,ty,lr)
    return nn

### 08. Tain The Neural Netwok
def trainNNModel(nn,iteration):
    nn.train(iteration) # Train the Neural Network

### 09. Plot cost vs iteration gaph
def plotCostVsIteration(nn):
    nn.plotCostFunction()

### 10. Predict using the trained Model
def predictUsingTheModel(my_data,nn):
    #print(nn.getFinalResult())
    print(deNormalizeY(my_data,nn.getFinalResult()))

### 00. Main Function
def main():
    print("Loading date from CSV file")
    my_data = loadData()
    print(my_data[:3])
    print(my_data.shape)
    
    print("Removing Missing Data")
    my_data = checkForMissingData(my_data)
    print(my_data[:3])
    print(my_data.shape)
    
    print("Normalizing Data")
    my_data_norm = normaLizeData(my_data)
    print(my_data_norm[:3])
    print(my_data_norm.shape)
    
    print("Checking De Normalization Function")
    tmpVal = deNormalizeY(my_data,0.35012337)
    print(tmpVal)
    
    print("Seperating X and Y")
    x,y = seperateXY(my_data_norm)
    print("X = ",x)
    print("Y = ",y)
    
    print("Seperating Train and Test Set")
    train_x,test_x,train_y,test_y = seperateTrainAndTestSet(x,y)
    print("Train X = ",train_x.shape)
    print("Test X = ",test_x.shape)
    print("Train Y = ",train_y.shape)
    print("Test Y = ",test_y.shape)
    
    print("Create Neural Network Structure")
    # a 1 layer neural network will have 1 Input layer + 1 Output Layer
    nn = createNNModel(1,np.array([train_x.shape[1],1]),train_x,train_y,0.1) # lc = Layer Count, nc = Node Count for rach layer, lr= = learning rate
    nn.describe() # Describe the Neural Network
    
    print("Train The Neural Network")
    trainNNModel(nn,2000) # neural network class and iteation
    nn.describe() # Describe the Neural Network
    
    print("Plot Cost Vs Iteration gaph")
    plotCostVsIteration(nn)
    
    print("Predict using the trained model")
    predictUsingTheModel(my_data,nn)

if __name__ == "__main__":
    main()
    
    
    
    
    




