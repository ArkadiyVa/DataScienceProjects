class AdaBoost():
    def __init__(self, iters=10, learning_rate=1):
        self.learning_rate = learning_rate
        self.iters = iters
        self.estimators = []
        self.y_preds = []
        self.estimators_weight = []
    
    def fit(self, X_train, y_train):
        sample_weight = np.ones(len(y_train))/len(y_train)
        for _ in range(self.iters):   
            #DecisionTreeClassifier classifier
            estimator = DecisionTreeClassifier(max_depth = 1, max_leaf_nodes=2)
            estimator.fit(X_train, y_train, sample_weight=sample_weight)
            y_predict = estimator.predict(X_train)
            self.estimators.append(estimator)

            estimator_error = ((y_predict != y_train).dot(sample_weight))/sum(sample_weight)
            estimator_weight =  self.learning_rate * np.log((1 - estimator_error)/estimator_error)
            #New sample weights
            sample_weight *= np.exp(estimator_weight * (y_predict != y_train))

            self.estimators_weight.append(estimator_weight.copy())   
            
    def prediction(self,X_test):
        
        for est in self.estimators:   
            y_predict = est.predict(X_test)
            self.y_preds.append(y_predict.copy())
            
        return ([np.sign((np.array(self.y_preds)[:,point] * self.estimators_weight).sum())\
                                                                     for point in range(len(X_test))])
    def get_estimators(self):
        return self.estimators
    
    def get_estimators_weight(self):
        return self.estimators_weight