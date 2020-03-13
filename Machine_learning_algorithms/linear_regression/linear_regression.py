import numpy as np

class LinearRegression:
    
    def __init__(self,Data,target): 
        ones = np.ones(Data.shape)
        Data = np.append(ones,Data,axis=1)
        self.Data = Data
        self.target = target
        self.n = Data.shape[0]
        self.theta = np.ones(Data.shape[1])
        
    def MSE(self):
        '''Cost function for linear regression (Mean square error)'''
        h=np.dot(self.Data,self.theta)
        self.loss=(1/(2*self.n))*np.sum((h-self.target)**2)
        return self.loss
    
    def GradientDescent(self,num_of_iter,lr):
        '''Gradient descent for linear regression (MSE)'''
        for i in range(num_of_iter):
            #Gradient for MSE
            h = np.dot(self.Data,self.theta)
            self.theta = self.theta-(lr/self.n)*self.Data.T.dot(h-self.target)
        return self.theta
    
    def OLS(self,X,y):
        '''Ordinary least squares for linear regression'''
        ones = np.ones(X.shape)
        X = np.append(ones,X,axis=1)
        inv = np.linalg.inv(np.dot(self.Data.T,self.Data))
        self.w = np.dot(np.dot(inv,self.Data.T),self.target)
        #prediction
        y_pred = np.dot(X,self.w)
        return y_pred,np.sum(abs(y-y_pred))/self.n
                   
    def predict(self,X,y):
        ones=np.ones(X.shape)
        X=np.append(ones,X,axis=1)
        self.target_pred=np.dot(X,self.theta)
        self.error_mae=np.sum(abs(self.target_pred-y))/self.n
        return self.target_pred,self.error_mae
                
    def returnTheta(self):
        return self.theta