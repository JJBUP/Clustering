import time

import torch
from sklearn import datasets
import matplotlib.pyplot as plt

# Plot the results
from clustering.k_means import k_means


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
    X1, _ = datasets.make_blobs(n_samples=n_samples, centers=100, random_state=random_state, cluster_std=1,
                                center_box=(100, 100))
    X2, _ = datasets.make_circles(n_samples=n_samples, factor=0.7, noise=.05, random_state=random_state)
    X3, _ = datasets.make_moons(n_samples=n_samples, noise=.1, random_state=random_state)
    # 模拟torch输入
    X1 = torch.tensor(X1).unsqueeze(dim=0)
    X2 = torch.tensor(X2).unsqueeze(dim=0)
    X3 = torch.tensor(X3).unsqueeze(dim=0)
    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    X = torch.concat([X1, X2, X3], dim=0).to(device)

    # 应用算法
    k = 50
    T1 = time.perf_counter()
    labels_pt, centers_pt = k_means(X, k)  # X:[B,N,K] ---> result [B,N]
    T2 = time.perf_counter()
    print("执行时间：", T2 - T1)
    labels_np, centers_np = labels_pt.cpu().numpy(), centers_pt.cpu().numpy()
    # 展示结果
    for i in range(labels_np.shape[0]):
        label = labels_np[i]
        center = centers_np[i]
        print(label.shape)
        print("label list:", [i for i in range(0, len(set(label)))])
        plot(X[i].cpu().numpy(), label, center, k)
