import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
    
class SupportVectorClustering():
    
    def __init__(self):
        pass
    
    def dataset(self,xs):
        self.xs = xs
        self.N = len(xs)
    def parameters(self,p=0.1, q=1):
        self.p = p
        self.q = q
        self.C = 1/(self.N*p)
    def kernel(self,x1,x2):
        return np.exp(-self.q*np.sum((x1-x2)**2,axis=-1))
    def kernel_matrix(self):
        self.km=np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                self.km[i,j] = self.kernel(self.xs[i],self.xs[j])
    def find_beta(self):
        beta = cvx.Variable(self.N)
        objective = cvx.Maximize(cvx.sum(beta)-cvx.quad_form(beta, self.km ))
        constraints = [0 <= beta,beta<=self.C,cvx.sum(beta)==1]
        prob = cvx.Problem(objective, constraints)
        result = prob.solve()
        self.beta = beta.value
    def r_func(self,x):
        return self.kernel(x,x)-2*np.sum([self.beta[i]*self.kernel(self.xs[i],x) for i in range(self.N)]) + self.beta.T@self.km@self.beta
    def sample_segment(self,x1,x2,r,n=10):
        adj = True
        for i in range(n):
            x = x1+(x2-x1)*i/(n+1)
            if self.r_func(x)>r:
                adj = False
                return adj
        return adj
    def cluster(self):
        svs_tmp = np.array(self.beta<self.C)*np.array(self.beta>10**-8)
        self.svs = np.where(svs_tmp==True)[0]
        bsvs_tmp = np.array(self.beta>=self.C)
        self.bsvs = np.where(bsvs_tmp==True)[0]
        self.r = np.mean([self.r_func(self.xs[i]) for i in self.svs[:5]])
        self.adj = np.zeros((self.N,self.N))
        for i in range(self.N):
            print(i)
            if i not in self.bsvs:
                for j in range(i,self.N):
                    if j not in self.bsvs:
                        self.adj[i,j]=self.adj[j,i]=self.sample_segment(self.xs[i],self.xs[j],self.r)
    def return_clusters(self):
        ids = list(range(self.N))
        self.clusters = {}
        num_clusters = 0
        while ids:
            num_clusters+=1
            self.clusters[num_clusters]=[]
            curr_id = ids.pop(0)
            queue = [curr_id]
            while queue:
                cid = queue.pop(0)
                for i in ids:
                    if self.adj[i,cid]:
                        queue.append(i)
                        ids.remove(i)
                self.clusters[num_clusters].append(cid)
            
                
    def show_plot(self):
        labels = np.zeros(self.xs.shape[0])
        for i in self.clusters.keys():
            for j in self.clusters[i]:
                labels[j] = int(i)
        
        from pandas import DataFrame
        from matplotlib import pyplot
        df = DataFrame(dict(x=xs[:,0], y=xs[:,1], label=labels))
        colors ={1:'r',2:'b',3:'g',4:'c',5:'m',6:'y',7:'k',8:'b',9:'c'}
        fig, ax = pyplot.subplots()
        grouped = df.groupby('label')
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
        pyplot.show()