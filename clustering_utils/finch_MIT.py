# -*- coding: utf-8 -*-
"""
FINCH - First Integer Neighbor Clustering Hierarchy Algorithm
"""
import time

# Author: Eren Cakmak <eren.cakmak@uni-konstanz.de>
#
# License: MIT

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import silhouette_score


class FINCH():
    """
     A class to perform the FINCH clustering
     """

    def __init__(self, metric='euclidean', n_jobs=1):
        """
        :param metric: string default='euclidean'
         The used distance metric - more options are
         ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
         ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’,
         ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘rogerstanimoto’, ‘sqeuclidean’,
         ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘yule’.
        :param n_jobs: cpu的进程数，帮助knn算法加速计算
        """
        self.metric = metric
        self.n_jobs = n_jobs

    def _finch(self, X, prev_n_clusters, prev_cluster_indices):
        """
        层次聚类，使用finch paper 的Eq1将近邻合并
        ----------
        :param X: 原始点数据 ndarray [N,D]
        :param prev_n_clusters: 上一次的聚类中心 ndarray [N,D]，第一次默认所有点均为聚类中心
        :param prev_cluster_indices: 上一次簇的所有点的索引，list[list]，第一次默认聚类中只有自身一个点
        :return:n_clusters_:当前联通数 int
                labels_:当前所有点的label ndarray
                cluster_centers:当前簇的质心 ndarray
                cluster_indices_:当前簇所有点的索引 list[list]
        """
        # -----核心算法------
        # 判断是否为第一次执行
        if prev_n_clusters is None:
            # 第一次执行将X作为data计算近邻
            data = X
        else:
            # 将先前簇的质心作为data计算近邻
            data = prev_n_clusters

        # 1. sklearn的近邻算法
        nbrs = NearestNeighbors(n_neighbors=2,  # 近邻k = 2，自身与最近点
                                metric=self.metric,
                                n_jobs=self.n_jobs).fit(data)

        # 返回邻接矩阵[n,n]（csr稀疏矩阵存储）
        connectivity = nbrs.kneighbors_graph(data)
        # 核心: 邻接矩阵 @ 邻接矩阵转置 = Eq.1 1(1.实现对称，2.带有对角线的邻接矩阵与转置的乘积能实现 Eq.1)
        connectivity @= connectivity.T
        # 删除对角线
        connectivity.setdiag(0)
        connectivity.eliminate_zeros()
        # 核心：计算有向图的连通性
        #   n_connected_components_:联通个数
        #   labels_标签，联通的节点标签相同
        n_clusters_, labels_ = connected_components(csgraph=connectivity)

        # 数据整理：
        # 1.获得新一轮cluster各点的标签
        # 第一轮簇的标签就是点的标签
        if len(labels_) < self.n_points:
            new_labels = np.full(self.n_points, 0)
            # 簇标签转为点的标签
            for i in range(n_clusters_):
                idx = np.where(labels_ == i)[0]
                idx = sum([prev_cluster_indices[j] for j in idx], [])  # 将label相等的簇(list)合并
                new_labels[idx] = i  # 更改合并后label的标签
            labels_ = new_labels
        # 数据整理：
        cluster_centers_ = []
        cluster_indices_ = []
        for i in range(n_clusters_):
            # 2.获得不同簇点的索引
            idx = np.where(labels_ == i)[0]
            cluster_indices_.append(idx.tolist())
            # 3.获得簇的质心
            xc_mean = X[idx].mean(axis=0)
            cluster_centers_.append(xc_mean)

        # n_connected_components_:总簇的个数 [S]
        # labels_:N个点的标签,ndarray [N]
        # cluster_centers_:当前簇中心,ndarray [S,D]
        # cluster_indices_:簇的索引 list[[簇0idx],[簇2idx],[簇3idx]]
        return n_clusters_, labels_, np.array(cluster_centers_), cluster_indices_

    def fit(self, X):
        """
        应用finch算法，实现多次聚类和指定聚类数目，返回【所有层次聚类得结果】,不计算轮廓系数
        ----------
        :param X: 原始点数据 ndarray [N,D]
        :return:result={{
                n_clusters:当前联通数 int
                labels:当前所有点的label ndarray
                cluster_centers:当前簇的质心 ndarray
                cluster_indices:当前簇所有点的索引 list[list]}
                   ...
                {......}

                }
        """

        self.n_points = X.shape[0]

        # 分区的结果
        results = {}

        # 中间结果
        cluster_centers_ = None
        cluster_core_indices_ = None

        n_clusters_ = len(X)

        print('FINCH Partitionings')
        print('-------------------')

        i = 0
        while n_clusters_ > 1:
            print("finch前cluster:", n_clusters_)
            T1 = time.perf_counter()
            n_clusters_, labels_, cluster_centers_, cluster_core_indices_ = self._finch(
                X, cluster_centers_, cluster_core_indices_)
            T2 = time.perf_counter()
            print("finch后cluster:", n_clusters_)
            print("执行时间:", T2 - T1)
            if n_clusters_ == 1:
                break
            else:
                print('Clusters in %s partition: %d' %
                      (i, n_clusters_))

            results['parition_' + str(i)] = {
                'n_clusters': n_clusters_,
                'labels': labels_,
                'cluster_centers': cluster_centers_,
                'cluster_indices': cluster_core_indices_
            }
            i += 1
        return results  # 返回所有聚类结果

    def fit_predict(self, X):
        """
        应用finch算法，实现多次聚类和指定聚类数目,【返回得分最好得label】

        :param X: 原始点数据 ndarray [N,D]
        :return:result={
                n_connected_components_:当前联通数 int
                labels_:当前所有点的label ndarray
                cluster_centers:当前簇的质心 ndarray
                cluster_indices_:当前簇所有点的索引 list[list]
                }

        """

        self.n_points = X.shape[0]

        # the results of the partitioning
        results = {}

        # intermediate results
        cluster_centers_ = None
        cluster_core_indices_ = None

        # min silhouette coefficent score
        max_sil_score = -1

        n_connected_components_ = len(X)

        print('FINCH Partitionings')
        print('-------------------')

        i = 0
        best_parition = {}
        while n_connected_components_ > 1:
            n_connected_components_, labels_, cluster_centers_, cluster_core_indices_ = self._finch(
                X, cluster_centers_, cluster_core_indices_)

            if n_connected_components_ == 1:
                break

            # in this version the silhouette coefficent is computed
            # 计算轮廓系数，选出最佳聚类簇
            sil_score = silhouette_score(X, labels_, metric=self.metric)
            results['parition_' + str(i)] = {
                'n_clusters': n_connected_components_,
                'labels': labels_,
                'cluster_centers': cluster_centers_,
                'cluster_core_indices': cluster_core_indices_,
                'silhouette_coefficient': sil_score
            }

            print(
                'Clusters in %s partition: %d with average silhouette score %0.2f'
                % (i, n_connected_components_, sil_score))

            if max_sil_score <= sil_score and i != 0:
                best_parition = results['parition_' + str(i)]
                max_sil_score = sil_score

            i += 1

        return best_parition
