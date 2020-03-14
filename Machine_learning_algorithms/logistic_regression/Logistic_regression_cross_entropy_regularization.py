class LogisticRegression_l2:
    
    def __init__(self, _lambda=0.001, lr=0.01, num_iter=10000, w0=0.5, verbose=True):
        self.lr = lr
        self.num_iter = num_iter
        self.verbose = verbose
        self._lambda = _lambda
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
                pred.append(0)
        return pred

    def cost(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean() + self._lambda/(2 * self.m) * sum(self.theta ** 2)

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = (1/self.m) * (np.dot(X.T, (h - y)) + self._lambda * self.theta)
            self.theta -= self.lr * gradient
            
            if (self.verbose) and (i % 100) == 0:
                loss = self.cost(h, y)
                y_pred = self.prediction(self.theta, X)
                
    def get_theha(self):
        return self.theta