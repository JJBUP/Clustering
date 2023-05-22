import time

import numpy as np
import torch
from sklearn import datasets
import matplotlib.pyplot as plt
from clustering_utils.finch_MIT import FINCH


# Plot the results
def plot(data, labels, centers, k):
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6',
              '#6a3d9a']

    for i in range(-1, len(set(labels))):
        if i == -1:
            col = [0, 0, 0, 1]
        else:
            col = colors[i % len(colors)]

        clust = data[np.where(labels == i)]
        plt.scatter(clust[:, 0], clust[:, 1], c=[col], s=1)
    for i, xy in enumerate(centers):
        plt.text(xy[0], xy[1], i)
    plt.show()

    return None


if __name__ == '__main__':
    n_samples = 7000
    n_clusters = 100
    random_state = 42
    X1, _ = datasets.make_blobs(n_samples=n_samples, n_features=2, centers=100, random_state=random_state)
    X2, _ = datasets.make_circles(n_samples=n_samples, factor=0.7, noise=.05, random_state=random_state)
    X3, _ = datasets.make_moons(n_samples=n_samples, noise=.1, random_state=random_state)

    X = np.vstack([X1.reshape(1, n_samples, -1), X3.reshape(1, n_samples, -1), X2.reshape(1, n_samples, -1)])
    # 应用算法
    fin = FINCH()
    paritions = []
    for i in torch.arange(X.shape[0]):
        T1 = time.perf_counter()
        paritions += [fin.fit_predict(X[i])] # fit_predict计算聚类轮廓系数得分，返回最佳
        T2 = time.perf_counter()
        print("执行时间：", T2 - T1)

    # 展示结果 show fit_predict
    for i, p in enumerate(paritions):
        labels = p['labels']
        centers = p['cluster_centers']
        print('label shape', labels.shape)
        print("label list:", [i for i in range(0, len(set(labels)))])
        plot(X[i], labels, centers, k=len(set(labels)))

