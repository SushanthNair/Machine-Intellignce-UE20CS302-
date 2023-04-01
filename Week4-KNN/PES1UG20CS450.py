import numpy as np
'''
SRN: PES1UG20CS450
Name: Sushanth M Nair
Section: H
Roll No: 50
'''

class KNN:
    """
    K Nearest Neighbours model
    Args:
        k_neigh: Number of neighbours to take for prediction
        weighted: Boolean flag to indicate if the nieghbours contribution
                  is weighted as an inverse of the distance measure
        p: Parameter of Minkowski distance
    """

    def __init__(self, k_neigh, weighted=False, p=2):

        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p

    def fit(self, data, target):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix( M data points with D attributes each)(float)
            target: Vector of length M (Target class for all the data points as int)
        Returns:
            The object itself
        """
        if(np.shape(data)[0] == np.shape(target)[0]):
            self.data = data
            self.target = target.astype(np.int64)

        return self
    def Minkowski(self,arr,p,data):
        y=[]
        n = np.shape(data)[1]
        for i in data[:,0:n]:
            m = pow(sum(pow(abs(a-b),p) for a, b in zip(arr, i)),1/p)
            y.append(m)
        return y
    def find_distance(self, x):
        """
        Find the Minkowski distance to all the points in the train dataset x
        Args:
            x: N x D Matrix (N inputs with D attributes each)(float)
        Returns:
            Distance between each input to every data point in the train dataset
            (N x M) Matrix (N Number of inputs, M number of samples in the train dataset)(float)
        """
        l=[]
        for i in x:
            m = self.Minkowski(i,self.p,self.data)
            l.append(m)
        
        return l

    def k_neighbours(self, x):
        """
        Find K nearest neighbours of each point in train dataset x
        Note that the point itself is not to be included in the set of k Nearest Neighbours
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            k nearest neighbours as a list of (neigh_dists, idx_of_neigh)
            neigh_dists -> N x k Matrix(float) - Dist of all input points to its k closest neighbours.
            idx_of_neigh -> N x k Matrix(int) - The (row index in the dataset) of the k closest neighbours of each input

            Note that each row of both neigh_dists and idx_of_neigh must be SORTED in increasing order of distance
        """
        
        l=[]
        a=[]
        b=[]
        c=[]
        l = self.find_distance(x)
        for i in l:
            o={}
            for j in i:
                o[j] = i.index(j)
            idx=[]
            dist=[]
            dist = np.sort(i)
            dist = dist[0:self.k_neigh]
            
            for k in dist:
                idx.append(o[k])
            a.append(dist)
            b.append(idx)
        c.append(a)
        c.append(b)
        
        return c


    def predict(self, x):
        """
        Predict the target value of the inputs.
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            pred: Vector of length N (Predicted target value for each input)(int)
        """
        l = self.k_neighbours(x)
        b=[]
        data1= self.target
        n = np.shape(self.data)[1]
        for i in l[1]:
            a=[]
            for j in i:
                a.append(data1[j])
            b.append(max(set(a), key = a.count))
        return b

    def evaluate(self, x, y):
        """
        Evaluate Model on test data using 
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix(float)
            y: True target of test data(int)
        Returns:
            accuracy : (float.)
        """
        b = self.predict(x)
        count = 0
        l = len(b)
        for (i,j) in list(zip(b,y)):
            if(i == j):
                count = count+1
        acc = (count/l)*100
        return acc
