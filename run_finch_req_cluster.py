import time

import torch
from sklearn import datasets
import matplotlib.pyplot as plt
from clustering.finch_req_cluster import FINCH


# Plot the results
def plot(data, labels, centers, k):
    # 生成随机颜色
    cmap = plt.cm.get_cmap('hsv', k)
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=1, cmap=cmap)
    for i, xy in enumerate(centers):
        plt.text(xy[0], xy[1], i)

    plt.show()

    return None


if __name__ == '__main__':
    n_samples = 5000
    n_clusters = 100
    random_state = 42
    X1, _ = datasets.make_blobs(n_samples=n_samples, centers=100, random_state=random_state, center_box=(100, 100))
    X2, _ = datasets.make_circles(n_samples=n_samples, factor=0.7, noise=.05, random_state=random_state)
    X3, _ = datasets.make_moons(n_samples=n_samples, noise=.1, random_state=random_state)
    # 模拟torch输入
    X1 = torch.tensor(X1).unsqueeze(dim=0)
    X2 = torch.tensor(X2).unsqueeze(dim=0)
    X3 = torch.tensor(X3).unsqueeze(dim=0)
    X = torch.concat([X1, X2, X3], dim=0)

    # 应用算法
    k = 50
    fin = FINCH()
    paritions = []
    T1 = time.perf_counter()
    for i in torch.arange(X.shape[0]):
        T1 = time.perf_counter()
        paritions += [fin.fit_req_cluster(X[i].numpy(), k)]
        T2 = time.perf_counter()
        print("执行时间：", T2 - T1)

    # 展示结果
    for i, p in enumerate(paritions):
        labels = p['labels']
        centers = p['cluster_centers']
        print(labels.shape)
        print("label list:", [i for i in range(0, len(set(labels)))])
        plot(X[i].numpy(), labels, centers, k)
