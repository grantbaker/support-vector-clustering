import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
plt.style.use('seaborn-whitegrid')

from scalable_support_vector_clustering import ScalableSupportVectorClustering

if __name__ == '__main__':
    ms = sklearn.datasets.make_moons(n_samples=500,noise=0.1)[0]
    cs = sklearn.datasets.make_circles(n_samples=800,noise=0.01)[0]



    X = sklearn.datasets.make_blobs(n_samples=500)[0]
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    aniso = np.dot(X, transformation)




    ssvc = ScalableSupportVectorClustering()

    ssvc.dataset(ms)
    # ssvc.dataset(cs)
    # ssvc.dataset(aniso)

    ssvc.parameters(p=0.002, B=300, q=7, eps1=0.03, eps2=10**-5, step_size=10**-1)
    ssvc.find_alpha(epochs=50)
    ssvc.cluster()

    ssvc.show_plot(figsize=(8,6))
    ssvc.show_bdd(xmin=-1.1,xmax=2.4,ymin=-1,ymax=1.5,n=50,figsize=(8,6))