import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

class ScalableSupportVectorClustering():
    
    def __init__(self):
        pass
    
    def dataset(self, xs):
        self.xs = xs
        
    def parameters(self,
                   p=0.1,
                   B=100,
                   q=0.1,
                   eps1=10**-4,
                   eps2=10**-1,
                   m=10,
                   step_size=10**-2):
        
        self.p = p
        self.N = self.xs.shape[0]
        self.C = 1/(self.N*p)
        self.B = B
        self.q = q
        self.eps1 = eps1
        self.eps2 = eps2
        self.m = m
        self.step_size= step_size
        
    def _kernel(self, x1, x2):
        return np.exp(-self.q*np.linalg.norm(x1 - x2, 2)**2)
    
    def find_alpha(self, epochs=1):
        self.alpha, self.indices = self._algo_2(xs=self.xs,
                                                K=self._kernel,
                                                C=self.C,
                                                B=self.B,
                                                q=self.q,
                                                epochs=epochs,
                                                step_size=self.step_size)
        
    def _algo_2(self, xs, K, C, B, q, epochs, step_size=10**-2):
        alpha = np.zeros(xs.shape[0])
        b = 0
        indices = set()
        t = 0
        for epoch in range(epochs):
            print("Epoch: ", epoch)
            
            # for partial results
            self.alpha, self.indices = alpha, indices
            
            for i in range(len(xs)):
                t += 1
#                 if i % 100 == 0: print('i =', i, 'of ', len(xs))
                n_t = np.random.choice(xs.shape[0])
                x_t = xs[n_t, :] # choose one randomly
                indicator = 0
                for j in indices:
                    indicator += alpha[j] * K(xs[j],x_t)
                if indicator < 1:
                    alpha = (i/(i+3))*alpha
                    alpha[n_t] += C*(step_size)
                    if n_t not in indices:
                        indices.add(n_t)
                        b += 1
                        if b > B:
                            p_t = -1
                            obj = 999999
                            for j in indices:
                                a = alpha[j]
                                if a < obj:
                                    p_t = j
                                    obj = a
                            b = B
                            alpha[p_t] = 0
                            indices.remove(p_t)
        return alpha, indices
    
    def cluster(self):
        self.clusters = self._algo_3(xs=self.xs,
                                     alpha=self.alpha,
                                     indices=self.indices,
                                     K=self._kernel,
                                     eps1=self.eps1,
                                     eps2=self.eps2,
                                     m=self.m)
    
    def _algo_3(self, xs, alpha, indices, K, eps1, eps2, m):
        def P(x):
            num=np.zeros(xs.shape[1])
            denom=0
            for j in indices:
                a = alpha[j] * K(xs[j], x)
                #print("asdf",j, a, alpha[j], K(xs[j], x))
                num += a * xs[j]
                denom += a
            #print(num, denom)
            return num/denom
        
        def f(x):
            s = -1.0
            for j in indices:
                s += alpha[j]*K(xs[j],x)
            return s
        
        def sample_segment(x1,x2):
            for i in range(1,m):
                x = x1+(x2-x1)*i/(m)
                if f(x) < 0: # 
                    return False
            return True
        
        def nearest_neighbor(x, xs_, inds):
            d = np.linalg.norm(x-xs_[inds[0],:])
            out = inds[0]
            for i in inds:
                d_tmp = np.linalg.norm(x-xs_[i,:])
                if d_tmp<d:
                    d = d_tmp
                    out = i
            return out
        
        decision_func_vals = np.apply_along_axis(lambda x : np.abs(f(x)),1,xs)
        print(np.min(decision_func_vals))
        B_eps_inds = np.where(decision_func_vals <= eps1)[0]
        B_n = B_eps_inds.shape[0]
        print('Epsilon points:', B_n)
        
        E = []
        N_eq = 0
        l = 0
        eps_eq_map = {}
        for ind in B_eps_inds:
            print('Processing epsilon point', l,'/', B_n)
            xk = xs[ind]
            x = np.zeros(xk.shape[0])
            while np.linalg.norm(x-xk) > eps2:
                x = xk
                xk = P(x)
            norms = [np.linalg.norm(xk-x_eq) for x_eq in E]
#             print(xk)
#             print(E)
#             print(norms)
            if len(E) == 0:
                E.append(xk)
                eps_eq_map[ind] = N_eq
                N_eq += 1
            else:
                if not (min(norms) < 10*eps2):
                    E.append(xk)
                    eps_eq_map[ind] = N_eq
                    N_eq += 1
                else:
                    eps_eq_map[ind] = norms.index(min(norms))
            l += 1
        
        print('Equilibrium points:', N_eq)
        print(E)
        adj = np.zeros((N_eq, N_eq))
        for i in range(N_eq):
            print('Finding adjacency:', i)
            for j in range(i, N_eq):
                adj[i,j] = adj[j,i] = sample_segment(E[i],E[j])
        
        print(adj)
        
        ids = list(range(N_eq))
        clusters_eqs = {}
        num_clusters = 0
        while ids:
            num_clusters += 1
            curr_id = ids.pop(0)
            queue = [curr_id]
            while queue:
                cid = queue.pop(0)
                for i in ids:
                    if adj[i,cid]:
                        queue.append(i)
                        ids.remove(i)
                clusters_eqs[cid] = num_clusters
        
        ind_to_cluster = {}
        clusters = {}
        for i in range(N_eq):
            clusters[i+1] = []
            
        for eps_ind in B_eps_inds:
#             print(eps_ind)
#             print(eps_eq_map)
#             print(clusters_eqs)
            cluster = clusters_eqs[eps_eq_map[eps_ind]]
#             clusters[cluster].append(eps_ind)
            ind_to_cluster[eps_ind] = cluster
        
        for i in range(xs.shape[0]):
            if i not in B_eps_inds:
                x = xs[i]
                close = nearest_neighbor(x, xs, B_eps_inds)
#                 print(close)
#                 print(ind_to_cluster[close])
                clusters[ind_to_cluster[close]].append(i)
            
#         print(clusters)
        clusters[0] = B_eps_inds
        
        self._meta_ = (B_eps_inds, ind_to_cluster, E)
    
        return clusters
    
    
    def show_plot(self, n=1, figsize=(5,4)):
        labels = np.zeros(self.xs.shape[0])
        for i in self.clusters.keys():
            for j in self.clusters[i]:
                labels[j] = int(i)
#         print(labels)
#         print(len(self.xs[:,0]))
#         print(len(self.xs[:,1]))
        
        from pandas import DataFrame
        from matplotlib import pyplot
        

        N = self.xs.shape[1]
        k = 0
        for i in range(N):
            for j in range(i+1,N):
                k += 1
                if k <= n:
                    print('\n x =', i, ' y =', j)
                    df = DataFrame(dict(x=self.xs[:,i], y=self.xs[:,j], label=labels))
                    colors ={0:'k',1:'r',2:'b',3:'g',4:'c',5:'m',6:'y',7:'k',8:'b',9:'c'}
                    fig, ax = pyplot.subplots(figsize=figsize)
                    grouped = df.groupby('label')
                    for key, group in grouped:
                        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors.get(key,'k'))
                    pyplot.show()
                    
    def show_bdd(self, minx=-1, maxx=1, miny=-1, maxy=1, n=20, figsize=(8,6)):
        
        def nearest_neighbor(x, xs_, inds):
            d = np.linalg.norm(x-xs_[inds[0],:])
            out = inds[0]
            for i in inds:
                d_tmp = np.linalg.norm(x-xs_[i,:])
                if d_tmp<d:
                    d = d_tmp
                    out = i
            return out
        
        B_eps_inds, ind_to_cluster, E = self._meta_
        
        labels = np.zeros(self.xs.shape[0])
        for i in self.clusters.keys():
            for j in self.clusters[i]:
                labels[j] = int(i)
#         print(labels)
#         print(len(self.xs[:,0]))
#         print(len(self.xs[:,1]))

        x = np.linspace(minx, maxx, n)
        y = np.linspace(miny, maxy, n)
        testpoints = np.transpose([np.tile(np.linspace(minx, maxx, n), n), np.repeat(np.linspace(miny, maxy, n), n)])
        nn = testpoints.shape[0]
        testpoint_labels = np.zeros(nn)
        for i in range(nn):
            x = testpoints[i]
            
            indicator = 0
            for j in self.indices:
                indicator += self.alpha[j] * self._kernel(self.xs[j],x)
            if indicator < 1:
                testpoint_labels[i] = 5
            else:
                close = nearest_neighbor(x, self.xs, B_eps_inds)
                testpoint_labels[i] = ind_to_cluster[close]
        
        from pandas import DataFrame
        
        df = DataFrame(dict(x=self.xs[:,0], y=self.xs[:,1], label=labels))
        df2 = DataFrame(dict(x=testpoints[:,0], y=testpoints[:,1], label=testpoint_labels))
        colors ={0:'k',1:'r',2:'b',3:'g',4:'c',5:'m',6:'y',7:'k',8:'b',9:'c'}
        fig, ax = plt.subplots(figsize=figsize)
        grouped = df.groupby('label')
        grouped2 = df2.groupby('label')
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors.get(key,'k'))
        for key, group in grouped2:
            group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors.get(key,'k'), s=0.5)
        E = np.array(E)
        plt.plot(E[:,0],E[:,1], '+y')
        plt.show()
        