class KNN:
    def __init__(self,X_train,y_train,K):
        self.X_train=X_train
        self.y_train=y_train
        self.K=K
        
    def predict(self,X):
        y_pred=np.array([])
        
        for each in X:
            #find euclidean distance for each element
            eucl_dist=np.sum((each-self.X_train)**2,axis=1)
            y_eucl_dist=np.concatenate((self.y_train.reshape(self.y_train.shape[0],1),
                                 eucl_dist.reshape(eucl_dist.shape[0],1)),axis=1)
            y_eucl_dist=y_eucl_dist[y_eucl_dist[:,1].argsort()]
            #find K nearest elements
            K_neighbours=y_eucl_dist[0:self.K]
            #predict is the most common of the elements
            y_pred = np.append(y_pred,stats.mode(K_neighbours[:,0])[0][0])
            
        return y_pred