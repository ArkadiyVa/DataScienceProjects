class LogisticRegression:
    
    def __init__(self, lr=0.01, num_iter=10000, w0=0.5, verbose=True):
        self.lr = lr
        self.num_iter = num_iter
        self.verbose = verbose
        self.w0 = w0
        self.m = y.shape[0]
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def prediction(self, w, Data):
        pred = []
        z = np.dot(Data,w)
        a = self.sigmoid(z)
        
        for i in range(0,len(a)):
            if (a[i] > self.w0): 
                pred.append(1)
            elif (a[i] <= self.w0):
                pred.append(-1)
        return pred

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            val = -np.multiply(y,z)
            f = -np.multiply(y,np.exp(val))/(1+np.exp(val))
            gradient = np.dot(X.T,f.T)
            self.theta -= self.lr * gradient.T
            
            if (self.verbose) and (i % 100) == 0:
                loss = np.sum(np.log(1+np.exp(val)))
                y_pred = self.prediction(self.theta, X)
                print('loss =',round(loss,3),' Training Accuracy',round(accuracy_score(y, y_pred)*100,1))
                
    def get_theha(self):
        return self.theta