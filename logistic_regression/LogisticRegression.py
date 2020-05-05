import numpy as np
'''
L'intera classe può essere sostituita con il seguente codice, che per giunta permette di avere più di due classi 

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1000, random_state=1, solver='lbfgs', multi_class='ovr')

'''


class LogisticRegressionGD(object):
    """
    Logistic regression con implementazione a gradiente discendente.
    """

    def __init__(self, eta=0.01, n_iter=1, random_state=1):
        self.eta=eta
        self.random_state=random_state
        self.n_iter=n_iter
    
    def sigmoid(self,X):
        return 1.0 / (1.0 + np.exp(-self.net_input(X)))

    def net_input(self, X):
        return np.dot(X,self.w_[1:]) + self.w_[0]
        
    def fit(self, X, y):
        #Creo i pesi:
        self.w_ = np.random.normal(loc=0,scale=0.1, size=X.shape[1] + 1)
        for _ in range(self.n_iter):
            self.w_[0] += self.eta*np.sum(y-self.sigmoid(X))
            self.w_[1:] += self.eta*X.T.dot(y-self.sigmoid(X))
        

    def predict(self,X):
        result = np.dot(X,self.w_[1:]) + self.w_[0]
        return np.where(result>0,1,0)


if __name__ == "__main__":
    from sklearn import datasets
    import os
    os.system('cls')
    iris = datasets.load_iris()
    X = iris.data[:,[2,3]]
    y = iris.target
    X = X[(y==0)|(y==1)]
    y = y[(y==0)|(y==1)]
    print(np.unique(y))
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X,y,random_state=1,stratify=y,test_size=0.3)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X_tr,y_tr)
    X_tr_std = sc.transform(X_tr)
    X_te_std = sc.transform(X_te)

    lr = LogisticRegressionGD()
    lr.fit(X_tr_std,y_tr)
    y_pred = lr.predict(X_te_std)
    
    from sklearn.metrics import accuracy_score
    print("Accuracy : %.3f" %accuracy_score(y_te, y_pred))

    from decision_regions import plot_decision_regions
    import numpy as np
    import matplotlib.pyplot as plt

    X_comb_std = np.vstack((X_tr_std,X_te_std))
    y_comb = np.hstack((y_tr,y_te))

    plot_decision_regions(X=X_comb_std,
        y=y_comb,
        classifier=lr)

    plt.xlabel('pl[std]')
    plt.ylabel('pw[std]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
