import torch
# Author: JJB

# batch k means by pytorch
def k_means(data, k, max_time=100):
    """
    data:聚类数据[B,N,D]
    k:聚类中心
    max_time = 最大迭代次数
    """
    B, N, D = data.shape
    device = data.device
    # 生成k个随机聚类中心
    init_idx = torch.randint(low=0, high=N, size=(k,)).to(device)  # 随机聚类中心[B,K]
    centers = data[..., init_idx, :]  # 随机选择k个起始点 [B,K,D]
    time = 0  # 循环次数
    last_label = 0  # 上一轮label
    label = torch.tensor([]).to(device)  # 本轮的label
    # 计算所有点到k个中心点的距离，选取距离最近的点融进该簇
    while (time < max_time):

        data_ = data.unsqueeze(1).repeat(1, k, 1, 1)  # 复制n份，[B,N,D]->[B,K,N,D]
        centers_ = centers.unsqueeze(2).repeat(1, 1, N, 1)  # 复制n份，[B,K,D]->[B,K,N,D]
        dist = torch.mean((data_ - centers_) ** 2, 3)  # 计算距离,[B,K,N]
        label = dist.argmin(1)  # 依据最近距离标记label[B,N]

        if torch.sum(label != last_label) == 0:  # label没有变化,跳出循环
            return label, centers
        last_label = label

        new_centers = torch.tensor([])
        # 分别计算batch中每个的聚类中心（因为batch中对应的聚类中心等是不同的）
        for i in range(B):  # 更新类别中心点，作为下轮迭代起始
            # 筛选label为i的点
            centers_one_batch = torch.tensor([]).to(device)
            # 按照聚类中心个数重新计算质心
            for j in range(k):
                cluster_point = data[i][label[i] == j]
                # 取质心作为中心点
                if j == 0:
                    # fix bug:有些情况下，上一轮计算的质心距离所有的点都很远(不如其他点近)，
                    # 此时没有点帮助该点重新计算新的质心，于是我们选择保持质心不变
                    centers_one_batch = cluster_point.mean(0).unsqueeze(0) if cluster_point.shape[0] != 0 else \
                        centers[i][j].unsqueeze(0)

                else:
                    centers_one_batch = torch.cat([centers_one_batch,
                                                   cluster_point.mean(0).unsqueeze(0)
                                                   # fix bug:有些情况下，上一轮计算的质心距离所有的点都很远(不如其他点近)，
                                                   # 此时没有点帮助该点重新计算新的质心，于是我们选择保持质心不变
                                                   if cluster_point.shape[0] != 0
                                                   else centers[i][j].unsqueeze(0)]
                                                  , 0)  # midpoint:[B,N,D]

            if i == 0:
                new_centers = centers_one_batch.unsqueeze(0)
            else:
                new_centers = torch.cat([new_centers, centers_one_batch.unsqueeze(0)], 0)  # new_centers:[B,N,D]
        centers = new_centers  # new_centers:[B,N,D]
        time += 1  # while

    return label, centers
