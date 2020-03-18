class Naive_Bayes:
    
    def __init__(self,train_data):
        self.train_data = train_data
        self.mu = train_data.groupby('target').mean().values
        self.std = train_data.groupby('target').std().values
        self.classes_prob = train_data.iloc[:,2].value_counts().values/len(train_data)

    def prediction(self,X): 
        scores = []
        for p in range(len(self.classes_prob)):
            scores.append(likelyhood(X[0],self.mu[p][0],self.std[p][0]) * \
                          likelyhood(X[1],self.mu[p][1],self.std[p][1]) * self.classes_prob[p])

        return np.argmax(scores)