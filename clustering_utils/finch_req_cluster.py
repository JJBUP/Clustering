# -*- coding: utf-8 -*-
"""
FINCH - First Integer Neighbor Clustering Hierarchy Algorithm
"""
# Author: JJB

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix, triu


class FINCH():
    """
     A class to perform the FINCH clustering
     """

    def __init__(self, metric='euclidean', n_jobs=-1):
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
                cluster_centers_:当前簇的质心 ndarray
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

    def _req_cluster(self, k, X, prev_n_clusters, prev_cluster_centers, prev_cluster_indices):
        """
        采用基于最近距离的聚类方法，将距离最近的n-k个点合并
            若n-k个点合并后恰剩下k个点则执行一次
            若n-k个点合并后大于k则继续合并，直到合并为k个点
        :param k: 指定聚类数目的k int
        :param X: 原始点数据 ndarray [N,D]
        :param prev_n_clusters: 上一次的聚类中心 ndarray [N,D]，第一次默认所有点均为聚类中心
        :param prev_cluster_indices: 上一次簇的所有点的索引，list[list]，第一次默认聚类中只有自身一个点
        :param prev_cluster_indices: 上一次簇的所有点的索引，list[list]
        :return:n_clusters_:当前联通数 int
                labels_:当前所有点的label ndarray
                cluster_centers:当前簇的质心 ndarray
                cluster_indices_:当前簇所有点的索引 list[list]
        """
        dist = euclidean_distances(prev_cluster_centers, prev_cluster_centers)  # NN
        dist_csr = triu(csr_matrix(dist), k=1)  # 提取上对角线
        diff = prev_n_clusters - k  # 待合并索引数量
        idxs_in_data = np.argpartition(dist_csr.data, diff)[:diff]  # 距离最近的点在data中的索引

        # 根据求得索引生成稀疏矩阵
        row = dist_csr.row[idxs_in_data]
        col = dist_csr.col[idxs_in_data]
        data = np.ones(len(idxs_in_data))
        connectivity = csr_matrix((data, (row, col)), shape=(prev_n_clusters, prev_n_clusters)).toarray()
        # 核心：计算有向图的连通性
        #   n_connected_components_:联通个数
        #   labels_标签，联通的节点标签相同
        n_clusters_, labels_ = connected_components(csgraph=connectivity, directed=False)

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

    def _init_result(self, X):
        """
        初始化result，防止当第一轮聚类时的联通区间数目小于k，导致result中无结果出现bug
        :param X:
        :return:
        """
        n_clusters_ = X.shape[0]
        labels_ = np.arange(n_clusters_)
        cluster_centers_ = X
        cluster_indices_ = np.arange(n_clusters_).reshape((n_clusters_, 1)).tolist()
        return {'parition_-1': {
            'n_clusters': n_clusters_,
            'labels': labels_,
            'cluster_centers': cluster_centers_,
            'cluster_indices': cluster_indices_
        }}

    def _get_true_centers(self, X, cluster_centers):
        """
        获得簇中心距离最近得点
        ----------------------------
        :param X: 原始点数据 ndarray [N,D]
        :param cluster_centers: 上一次簇的质心 ndarray
        """
        nbrs = NearestNeighbors(n_neighbors=1,  # 近邻k = 1，质心不在数据当中，所以不用担心取到自己
                                metric=self.metric,
                                n_jobs=self.n_jobs).fit(X)
        # 返回邻接矩阵[n,n]（csr稀疏矩阵存储）
        # a,b = nbrs.kneighbors(cluster_centers)
        new_centers_idx = nbrs.kneighbors(cluster_centers, return_distance=False).reshape(-1)  # [N,K]-->[N*K]
        cluster_centers = X[new_centers_idx]
        return cluster_centers

    def fit_req_cluster(self, X, k, true_center=False):
        """
        应用finch算法，实现多次聚类和指定聚类数目
        ----------
        :param k: 指定聚类数目的k int
        :param X: 原始点数据 ndarray [N,D]
        :param true_center: 是否获得真实得质心 bool
        :return:result={
                n_clusters_:当前联通数 int
                labels_:当前所有点的label ndarray
                cluster_centers:当前簇的质心 ndarray
                cluster_indices_:当前簇所有点的索引 list[list]
                }

        """
        # check if input is correct

        self.n_points = X.shape[0]

        # 中间结果
        cluster_centers_ = None
        cluster_indices_ = None
        # 初始簇数目等于N数目
        n_clusters_ = len(X)

        # print('FINCH Partitionings')
        # print('-------------------')

        i = 0
        results = self._init_result(X)
        while n_clusters_ > k:
            # 近邻点划分簇
            n_clusters_, labels_, cluster_centers_, cluster_indices_ = self._finch(
                X, cluster_centers_, cluster_indices_)
            # 判断是否小于指定分区k，若小于则停止
            if n_clusters_ <= k:
                prev_praition = results['parition_' + str(i - 1)]
                n_clusters_, labels_, cluster_centers_, cluster_indices_ = self._req_cluster(
                    k, X, prev_praition['n_clusters'], prev_praition['cluster_centers'],
                    prev_praition['cluster_indices'])

            results['parition_' + str(i)] = {
                'n_clusters': n_clusters_,
                'labels': labels_,
                'cluster_centers': cluster_centers_,
                'cluster_indices': cluster_indices_
            }
            # print('Clusters in %s partition: %d' % (i, n_connected_components_))
            i += 1

        parition = results['parition_' + str(i - 1)]
        if true_center:
            true_cluster_centers = self._get_true_centers(X, parition['cluster_centers'])
            parition['cluster_centers'] = true_cluster_centers
        # 返回最后一次(刚刚大于等于k的聚类信息)
        return parition
