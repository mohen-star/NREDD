import os

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform, cdist
import libtiff as TIFF


def read_tif(tif_path):
    images = []
    tif = TIFF.TIFF.open(tif_path, mode='r')
    for image in tif.iter_images():
        # image = np.flipud(image)
        images.append(image)
    return np.array(images)


def save_tif(img, path):
    # if type == 'img':
    tif = TIFF.TIFF.open(path, mode='w')
    num = img.shape[0]
    for i in range(num):
        # img[i] = np.flipud(img[i])
        tif.write_image(((img[i]).astype(np.uint8)), compression=None)
    tif.close()
    return


def loadswc(filepath):
    # load swc file as a N X 7 numpy array
    swc = []
    with open(filepath) as f:
        lines = f.read().split("\n")
        for l in lines:
            if not l.startswith('#'):
                cells = l.split(' ')
                # remove empty units
                if len(cells) != 9:
                    for kk in range(len(cells) - 1, -1, -1):
                        if cells[kk] == '':
                            cells.pop(kk)
                if len(cells) != 7:
                    for i in range(7, len(cells)):
                        # print(i)
                        cells.pop()
                if len(cells) == 7:
                    cells = [float(c) for c in cells]  # transform string to float
                    swc.append(cells[0:7])
    return np.array(swc)


def saveswc(filepath, swc):
    if swc.shape[1] > 7:
        swc = swc[:, :7]
    # print(filepath)
    with open(filepath, 'w') as f:
        for i in range(swc.shape[0]):
            # print(swc[i, :])
            print('%d %d %.3f %.3f %.3f %.3f %d' %
                  tuple(swc[i, :].tolist()), file=f)


def find_child_idx(swc, node_idx0):
    return np.where(swc[:, 6] == node_idx0)[0]


def find_parent_idx(swc, node_idx6):
    return np.where(swc[:, 0] == node_idx6)[0]


def resample_swc_(swc):
    # swc = loadswc(swc_path)
    if swc.shape[1] < 8:
        axis7 = np.zeros((swc.shape[0], 1))
        swc = np.concatenate((swc, axis7), axis=1)
    else:
        swc[:, 7] = 0
    # print(swc[:, 7])
    swc_copy = swc.copy()

    # print(swc_copy)
    for i in range(swc.shape[0]):
        index0 = swc[i, 0]  # 节点标号
        index6 = swc[i, 6]  # 其父节点标号
        swc[i, 0] = i + 1  # 将节点标号按顺序重置
        # print(index0)
        # a7 = swc[i, 7]
        ch_index = find_child_idx(swc_copy, index0)
        # print(ch_index)
        for j in ch_index:
            swc[j, 6] = i + 1  # 修改其子节点对应父节点位置的标号
            swc[j, 7] = -1  # 标记为修改

        if swc[i, 6] != -1 and swc[i, 7] != -1:
            pa_index = find_parent_idx(swc_copy, index6)
            if len(pa_index) == 0:
                swc[i, 6] = -1
    return swc[:, :7]


def generate_sphere(Ma, Mp):
    # generate 3d sphere
    m1 = np.arange(1, Ma + 1, 1).reshape(-1, Ma)
    m2 = np.arange(1, Mp + 1, 1).reshape(-1, Mp)
    alpha = 2 * np.pi * m1 / Ma
    phi = -(np.arccos(2 * m2 / (Mp + 1) - 1) - (np.pi))
    xm = (np.cos(alpha).reshape(Ma, 1)) * np.sin(phi)
    ym = (np.sin(alpha).reshape(Ma, 1)) * np.sin(phi)
    zm = np.cos(phi)
    zm = np.tile(zm, (Mp, 1))
    sphere_core = np.concatenate([xm.reshape(-1, 1), ym.reshape(-1, 1), zm.reshape(-1, 1)],
                                 axis=1)  # y_axis=alpha[0:Ma],x_axis=phi[0:Mp]
    return sphere_core  # , alpha, phi


def get_edge_for_coonect(swc):
    swc_adj = np.zeros((swc.shape[0], swc.shape[0])).astype(np.float32)
    for i in range(swc.shape[0]):
        cur_point = swc[i, 0].astype(np.int32)
        par_point = swc[i, -1].astype(np.int32)
        if par_point == -1:
            swc_adj[cur_point - 1, cur_point - 1] = 1
        else:
            swc_adj[cur_point - 1, par_point - 1] = 1
            swc_adj[cur_point - 1, cur_point - 1] = 1
        cld_list = np.where(swc[:, -1] == cur_point)[0]
        for cld_idx in cld_list:
            cld_point = swc[cld_idx, 0].astype(np.int32)
            swc_adj[cur_point - 1, cld_point - 1] = 1
    adj_org = swc_adj
    # dis_edge = compute_adjacency_matrix_with_angle(swc, sigma=8.0)
    # # print(dis_edge[0])
    # swc_adj = np.maximum(swc_adj, dis_edge)
    D = np.power(np.sum(swc_adj, axis=-1), -0.5)
    D[np.isinf(D)] = 0
    D = np.diag(D)
    A = swc_adj.dot(D).transpose().dot(D)
    return adj_org, A


def calculate_angle_cos(A, B, C):
    """
    计算由三个点 A、B、C 定义的角度 ∠BAC，其中 A 是顶点。
    参数：
    A, B, C：三维空间中的三个点坐标（numpy 数组或列表）
    返回：
    余弦值
    """
    # 将输入转换为 numpy 数组
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    C = np.array(C, dtype=np.float64)

    # 计算向量 AB 和 AC
    AB = B - A
    AC = C - A

    # 计算向量的点积
    dot_product = np.dot(AB, AC)

    # 计算向量 AB 和 AC 的模长
    magnitude_AB = np.linalg.norm(AB)
    magnitude_AC = np.linalg.norm(AC)

    # 防止模长为零导致除零错误
    if magnitude_AB == 0 or magnitude_AC == 0:
        # print("无效输入，至少一个向量的长度为零：")
        # print(f"A: {A}, B: {B}, C: {C}")
        return 0

    # 计算夹角的余弦值
    cos_theta = dot_product / (magnitude_AB * magnitude_AC)

    # 限制余弦值在 -test_dataset 到 test_dataset 之间，以防浮动误差
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return cos_theta


def Spherical_Patches_Extraction(img2, position, SP_N, SP_core, SP_step=1):
    x = position[0]
    y = position[1]
    z = position[2]
    radius = 1
    j = np.arange(radius, SP_N * SP_step + radius, SP_step).reshape(-1, SP_N)
    ray_x = x + (SP_core[:, 0].reshape(-1, 1)) * j
    ray_y = y + (SP_core[:, 1].reshape(-1, 1)) * j
    ray_z = z + (SP_core[:, 2].reshape(-1, 1)) * j

    Rray_x = np.clip(np.rint(ray_x).astype(int), 0, 255)
    Rray_y = np.clip(np.rint(ray_y).astype(int), 0, 255)
    Rray_z = np.clip(np.rint(ray_z).astype(int), 0, 255)

    Spherical_patch_temp = img2[Rray_z, Rray_x, Rray_y]
    Spherical_patch = Spherical_patch_temp[:, 1:SP_N]

    SP = np.asarray(Spherical_patch)
    return SP


def change_coord(swc_path, mode='to_255'):
    swc_name = os.path.basename(swc_path).replace('.swc', '')
    # print(swc_name)
    xyz = swc_name.split('_')
    # print(xyz)
    x_cord = float(xyz[0].replace('x', ''))
    y_cord = float(xyz[1].replace('y', ''))
    z_cord = float(xyz[2].replace('z', ''))
    # print(swc_name)
    swc = loadswc(swc_path)
    if mode == 'to_255':
        if np.any(swc[:, 2:5] >= 256):
            swc[:, 2] = swc[:, 2] - x_cord
            swc[:, 3] = swc[:, 3] - y_cord
            swc[:, 4] = swc[:, 4] - z_cord
    elif mode == 'to_origin':
        if np.all(swc[:, 2:5] <= 256):
            swc[:, 2] = swc[:, 2] + x_cord
            swc[:, 3] = swc[:, 3] + y_cord
            swc[:, 4] = swc[:, 4] + z_cord
    return swc


def feature_normal_image(feature_tif):
    if feature_tif.max() - feature_tif.min() > 0:
        feature_tif = (feature_tif - feature_tif.min()) / (feature_tif.max() - feature_tif.min())
    feature_tif = np.power(feature_tif, 1.0)
    return feature_tif


def norm_feature_xyz(feature):
    feature_min = np.min(feature, axis=0)
    feature_max = np.max(feature, axis=0)

    range_ = feature_max - feature_min
    range_[range_ == 0] = 1

    normalized_feature = (feature - feature_min) / range_

    feature = normalized_feature
    return feature


def create_SPfeature(swc, tif, Ma=16, Mp=16, SP_N=10, SP_step=1):
    SP_core = generate_sphere(Ma, Mp)
    img_normalized = tif / 255.0
    gamma = 1.0
    img_gamma = np.power(img_normalized, gamma)
    img = np.uint8(img_gamma * 255)
    tif = img

    tif_list = []
    for i in range(swc.shape[0]):
        xyz = swc[i, 2:5]
        yxz = xyz[[1, 0, 2]]
        Spherical_patch = Spherical_Patches_Extraction(tif, yxz, SP_N, SP_core, SP_step)
        SP = Spherical_patch.reshape([Ma, Mp, SP_N - 1]).transpose([2, 0, 1])
        tif_list.append(SP)
    tif_list_a = np.array(tif_list).astype(np.float32)
    return tif_list_a


def swc_progress(swc, swc_path, Ma=16, Mp=16, SP_N=10, SP_step=1):
    adj_org, adj = get_edge_for_coonect(swc)
    feature = swc[:, 2:5]
    feature = norm_feature_xyz(feature)
    tif_path = swc_path.replace('.swc', '.tif')
    tif = read_tif(tif_path)

    feat_sp = create_SPfeature(swc, tif, Ma, Mp, SP_N, SP_step)
    feat_sp = feature_normal_image(feat_sp)
    feat_sp = torch.from_numpy(feat_sp).float()

    feat_xyz = torch.from_numpy(feature).float()
    adj_org = torch.from_numpy(adj_org).float()
    adj = torch.from_numpy(adj).float()
    return feat_xyz, feat_sp, adj, adj_org


def cal_ang_weight(idx1, idx2, swc):
    pointA = swc[idx1, 2:5]
    pointB = swc[idx2, 2:5]
    parent_A = find_parent_idx(swc, swc[idx1, 6])
    child_A = find_child_idx(swc, swc[idx1, 0])
    cos1_1 = 0
    cos1_2 = 0
    if len(parent_A) == 1 and swc[idx1, 6] != -1:
        pointC = swc[parent_A[0], 2:5]
        cos1_1 = (1 - calculate_angle_cos(pointA, pointB, pointC)) / 2

    if len(child_A) == 1:
        pointC = swc[child_A[0], 2:5]
        cos1_2 = (1 - calculate_angle_cos(pointA, pointB, pointC)) / 2

    cos1 = max(cos1_1, cos1_2)

    parent_B = find_parent_idx(swc, swc[idx2, 6])
    child_B = find_child_idx(swc, swc[idx2, 0])
    cos2_1 = 0
    cos2_2 = 0
    if len(parent_B) == 1 and parent_B != -1:
        pointD = swc[parent_B[0], 2:5]
        cos2_1 = (1 - calculate_angle_cos(pointB, pointA, pointD)) / 2
        # print('cos2_1 = {}'.format(cos2_1))
    if len(child_B) == 1 or (len(child_B) == 2 and len(parent_B) == 0):
        pointD = swc[child_B[0], 2:5]
        cos2_2 = (1 - calculate_angle_cos(pointB, pointA, pointD)) / 2

    cos2 = max(cos2_1, cos2_2)

    return 2 * cos1 * cos2 / (cos1 + cos2)


def Rec_adj(adj_recon, adj_org, swc, Max_distance=20, lamd_dist=0.5):
    adj_org = adj_org.cpu().numpy().astype(int)
    adj_recover = adj_org.copy()

    adj_recon = adj_recon.cpu().numpy()
    adj_pre = adj_recon.copy()
    index = np.where(np.sum(adj_org, axis=1) == 2)[0]  # 自身+一个连接

    # 删除边界端点
    if len(index) > 0:
        # 提取坐标并检查条件
        coordinates = swc[index, 2:5]
        mask = np.all((coordinates > 12) & (coordinates < 244), axis=1)  # 按行判断所有坐标是否满足
        index = index[mask]  # 从 index1 中筛选符合条件的索引
    else:
        print("index1 为空，无法进一步筛选")

    if len(index) == 0:
        return adj_recover

    # 对于端点它最好只连端点，但也有其他情况
    mark3 = np.zeros_like(adj_pre)
    mark3[:, index] = 1

    # 距离权重, 由于成像时，z轴分辨率更低，所以对于z轴方向的距离进行加权，使得其能够更好的区分z方向的连接关系
    distances = np.linalg.norm((swc[:, 2:5] - swc[index, 2:5][:, np.newaxis, :]) * (1, 1, 2),
                               axis=2)  # [len(index),len(swc)]

    radius = np.tile(swc[:, 5], (len(index), 1))
    radius[radius < 2] = 0
    distances = distances - radius
    distances[distances < 0] = 0
    distances[distances > Max_distance] = 0
    dis_w = 1 / (1 + np.exp(distances / Max_distance / lamd_dist - 1))
    dis_w[distances == 0] = 0

    mask_dis = np.ones_like(adj_pre)
    mask_dis[index] = (distances > 0)
    adj_pre = adj_pre * mask_dis

    ang_w = np.zeros_like(adj_pre)

    # 角度权重
    for idx_x in index:
        idx_y_list = np.where(adj_pre[idx_x] > 0)[0]
        # 删除已经与端点连接的点
        for idx_y in idx_y_list:
            if find_link(idx_x, idx_y, swc):
                adj_pre[idx_x][idx_y] = 0
                continue
            # 计算角度权重
            ang_w[idx_x, idx_y] = cal_ang_weight(idx_x, idx_y, swc)
            idx_x_1 = np.where(index == idx_x)[0][0]
            signal_wight = adj_pre[idx_x, idx_y] * 0.4 + dis_w[idx_x_1, idx_y] * 0.2 + ang_w[idx_x, idx_y] * 0.4
            if (distances[idx_x_1, idx_y] < Max_distance / 2 + 4 and ang_w[idx_x, idx_y] >= 0.8 and (adj_pre[
                                                                                                         idx_x, idx_y] >= 0.4 or signal_wight >= 0.55)) or \
                    distances[idx_x_1, idx_y] <= Max_distance / 4:
                mark3[idx_x, idx_y] = 1

    mark2 = np.ones_like(adj_pre).astype(np.float32)
    mark2[adj_pre == 0] = 0
    adj_pre = adj_pre * mark2

    # 计算综合概率
    adj_pre[index] = (adj_pre[index] * 0.4 + dis_w * 0.2 + ang_w[index] * 0.4) * mark3[index, :]

    k = 3
    idx_topk = np.argsort(adj_pre[index])[:, -k:][:, ::-1]
    values_topk = np.take_along_axis(adj_pre[index], idx_topk, axis=1)

    # 选择概率最大的点进行连接
    remain_nodes = list(index.copy())
    while values_topk.max() > 0:
        idx_x, idx_y = np.unravel_index(np.argmax(values_topk), values_topk.shape)
        node_x = index[idx_x]
        node_y = idx_topk[idx_x, idx_y]
        if adj_pre[node_x, node_y] != values_topk[idx_x, idx_y]:
            values_topk[idx_x, idx_y] = adj_pre[node_x, node_y]
            continue
        node_value = adj_pre[node_x, node_y]
        # print("3: ", values_topk[0])
        if adj_recover[node_x].sum() <= 2:
            # print('node: ', node_x, node_y)
            # print(idx_topk[idx_x])
            # print(values_topk[idx_x])
            # print('remain_nodes: ', remain_nodes)
            # 所有点的概率都不满足连接条件
            if node_value <= 0.45:
                break
            # 一个点最多3条边
            if adj_recover[node_y].sum() >= 4:
                # print(node_y, ': eage > 4')
                adj_pre[node_x, node_y] = 0
                values_topk[idx_x, idx_y] = 0
                continue

            # print("connet: ", node_x, node_y)
            adj_recover[node_x, node_y] = 1
            adj_recover[node_y, node_x] = 1

            remain_nodes.remove(node_x)
            adj_pre[node_x] = 0
            values_topk[idx_x] = 0

            adj_pre[remain_nodes, node_x] /= 2
            adj_pre[remain_nodes, node_y] /= 2

            if node_y in remain_nodes:
                remain_nodes.remove(node_y)
                adj_pre[node_y] = 0
                values_topk[np.where(index == node_y)[0][0]] = 0
    return adj_recover


def find_link(point1, point2, swc):
    if point1 == point2:
        return True
    parent_idx = point1
    # print(point1,point2)
    # parent_idx_list.append(parent_idx)
    while swc[parent_idx, 6] != -1:
        # print('swc.shape', swc.shape)
        parent_idx = np.where(swc[:, 0] == swc[parent_idx, 6])[0]
        # print(parent_idx)
        if parent_idx == point2:
            return True
        # # print('xxxx2')
        # # print('parent_idx', parent_idx)
        # # print('xxx: ', swc[parent_idx, 6])
        # if np.isin(parent_idx, parent_idx_list):
        #     break
        # parent_idx_list.extend(parent_idx)
    child_idx = list(np.where(swc[:, 6] == swc[point1, 0])[0])
    # child_idx_list = []
    while len(child_idx) > 0:
        cur_child = child_idx.pop()
        if cur_child == point2:
            return True
        # child_idx_refresh = []
        # if np.isin(child_idx[0], child_idx_list):
        #     # print(child_idx_list)
        #     break
        # child_idx_list.extend(child_idx)
        # print('swc.shape:', swc.shape)
        # print('child_idx_list:', child_idx_list)
        cur_child_child_idx = list(np.where(swc[:, 6] == swc[cur_child, 0])[0])
        while len(cur_child_child_idx) > 0:
            child_idx.insert(0, cur_child_child_idx.pop())
            # next_idx = np.where(swc[:, 6] == swc[idx, 0])[0]
            # # child_idx_list.extend(next_idx)
            # child_idx_refresh.extend(next_idx)
        # child_idx = child_idx_refresh

        # print('child_idx:', child_idx)
    # parent_idx_list.extend(child_idx_list)
    # # print(parent_idx_list)
    # if np.isin(point2, parent_idx_list):
    #     return True
    return False


class Node(object):
    def __init__(self, position, radius, node_type=3):
        self.position = position
        self.radius = radius
        self.node_type = node_type
        self.nbr = []


def get_undiscover(dist):
    for i in range(dist.shape[0]):
        if dist[i] == 100000:
            return i
    return -1


def compute_trees(n0):
    n0_size = len(n0)
    treecnt = 0
    q = []
    n1 = []
    dist = np.ones([n0_size, 1], dtype=np.int32) * 100000
    nmap = np.ones([n0_size, 1], dtype=np.int32) * -1  # index in output tree n1
    parent = np.ones([n0_size, 1], dtype=np.int32) * -1  # parent index in current tree n0
    # print('Search for Soma')
    for i in range(n0_size):
        if n0[i].node_type == 10000:
            q.append(i)
            dist[i] = 0
            nmap[i] = -1
            parent[i] = -1
            n0[i].node_type = 2
    # BFS
    while len(q) > 0:
        curr = q.pop(0)
        # print('curr_node:', curr + test_dataset)
        n = Node(n0[curr].position, n0[curr].radius, treecnt + 2)
        if parent[curr] >= 0:
            n.nbr.append((nmap[parent[curr]]))
            # print(len(n1) + test_dataset, curr + test_dataset, int(parent[curr] + test_dataset))
        n1.append(n)
        nmap[curr] = len(n1)
        for j in range(len(n0[curr].nbr)):
            adj = n0[curr].nbr[j]
            if dist[adj] == 100000:
                dist[adj] = dist[curr] + 1
                parent[adj] = curr
                # print(adj,curr)
                q.append(adj)

    while get_undiscover(dist) >= 0:
        treecnt = treecnt + 1
        seed = get_undiscover(dist)
        dist[seed] = 0
        nmap[seed] = -1
        parent[seed] = -1
        q.append(seed)
        while len(q) > 0:
            curr = q.pop(0)
            # print('curr_node:', curr + test_dataset)
            n = Node(n0[curr].position, n0[curr].radius, treecnt + 2)
            if parent[curr] >= 0:
                n.nbr.append((nmap[parent[curr]]))
                # print(len(n1) + test_dataset, curr + test_dataset, int(parent[curr] + test_dataset))
            n1.append(n)
            nmap[curr] = len(n1)
            for j in range(len(n0[curr].nbr)):
                adj = n0[curr].nbr[j]
                if dist[adj] == 100000:
                    dist[adj] = dist[curr] + 1
                    parent[adj] = curr
                    # print(adj, curr, nmap[parent[adj]])
                    q.append(adj)
    # print(len(n1))
    return n1


def build_nodelist(tree):
    _data = np.zeros((1, 7))
    cnt_recnodes = 0
    for i in range(len(tree)):
        if len(tree[i].nbr) == 0:
            cnt_recnodes += 1
            pid = -1
            new_node = np.asarray([cnt_recnodes,
                                   tree[i].node_type,
                                   tree[i].position[0],
                                   tree[i].position[1],
                                   tree[i].position[2],
                                   tree[i].radius,
                                   pid])
            _data = np.vstack((_data, new_node))

        else:
            # if len(tree[i].nbr) > 2:
            #    print(len(tree[i].nbr))
            for j in range(len(tree[i].nbr)):
                cnt_recnodes += 1
                pid = tree[i].nbr[j].squeeze()
                new_node = np.asarray([cnt_recnodes,
                                       tree[i].node_type,
                                       tree[i].position[0],
                                       tree[i].position[1],
                                       tree[i].position[2],
                                       tree[i].radius,
                                       pid])
                _data = np.vstack((_data, new_node))
    _data = _data[1:, :]
    return _data


def create_node_from_swc(swc, max_idx=None):
    swc_temp = swc.copy()
    node_list = []
    for i in range(swc_temp.shape[0]):
        node = Node(swc_temp[i, 2:5], swc_temp[i, 5], swc[i, 1])
        if i == 0:
            node = Node(swc_temp[i, 2:5], swc_temp[i, 5], node_type=10000)
        if max_idx is not None:
            idx = np.where(max_idx[:, 0] == i)
            if len(idx) >= 1:
                node.nbr = max_idx[idx][:, 1]
        node_list.append(node)
    return node_list


def get_swc_from_edges(adj, adj_org, swc, save_path, is_start=False):
    check_swc = swc.copy()
    adj = torch.sigmoid(adj)
    adj_rec = Rec_adj(adj, adj_org, check_swc)
    recovered = adj_rec
    np.fill_diagonal(recovered, 0)
    max_idx = np.where(recovered == 1)
    indices = np.column_stack((max_idx[0], max_idx[1]))

    node_list = create_node_from_swc(check_swc, indices)
    node_list = compute_trees(node_list)

    check_swc = build_nodelist(node_list)
    repair_path = save_path.replace('.swc', '._repair.swc')
    saveswc(repair_path, check_swc)
    # check_swc = change_coord(repair_path, mode='to_origin')
    # saveswc(save_path, check_swc)


def bulid_subgraphs(swc, deeplayer=4):
    Tree = generate_Ntree(swc)
    subtree_list = []
    for i in range(len(Tree)):
        cur_stree = []
        cur_node_list = []
        next_node_list = []
        cur_node_list.append(i)
        # print("start node: ", cur_node_list, Tree[i].nbr)
        for j in range(deeplayer):
            while (len(cur_node_list) > 0):
                cur_node = cur_node_list.pop()
                cur_stree.append(cur_node)
                # print('cur_node: ', cur_node)
                for nbr_node_id in Tree[cur_node].nbr:
                    if nbr_node_id not in cur_stree and swc[nbr_node_id][1] != -1:
                        # print("nbr: ", nbr_node_id)
                        next_node_list.append(nbr_node_id)
            cur_node_list = next_node_list.copy()
            next_node_list.clear()
        if len(cur_stree) > 16:
            cur_stree = cur_stree[:16]
        cur_stree1 = [Tree[idx] for idx in cur_stree]
        subtree_list.append(bulid_swc_feat(cur_stree1))
    return subtree_list


class Node1(object):
    def __init__(self, idx, position, radius, node_type=3):
        self.position = position
        self.radius = radius
        self.node_type = node_type
        self.nbr = []
        self.idx = idx


def generate_Ntree(swc):
    Node_list = []
    for i in range(0, swc.shape[0]):
        node = Node1(idx=i, position=swc[i, 2:5], radius=swc[i, 5], node_type=swc[i, 1])
        if swc[i, -1] != -1:
            p_id = np.where(swc[:, 0] == swc[i, -1])[0]
            # print(i,p_id)
            if len(p_id) >= 0:
                node.nbr.append(p_id[0])
        c_id = np.where(swc[i, 0] == swc[:, -1])[0]
        for j in range(len(c_id)):
            node.nbr.append(c_id[j])
        Node_list.append(node)
    return Node_list


def bulid_swc_feat(subtree):
    subtree_len = len(subtree)
    adj_raw = np.zeros([subtree_len, subtree_len])
    swc_feat = np.zeros([subtree_len, 3])
    idx_list = [subtree[i].idx for i in range(subtree_len)]
    for i in range(subtree_len):
        swc_feat[i] = subtree[i].position / 255.0
        adj_raw[i, i] = 1
        for nbr in subtree[i].nbr:
            if nbr not in idx_list:
                continue
            j = idx_list.index(nbr)
            adj_raw[j, i] = 1
            adj_raw[i, j] = 1

    D = np.power(np.sum(adj_raw, axis=-1), -0.5)
    D[np.isinf(D)] = 0
    D = np.diag(D)
    A = adj_raw.dot(D).transpose().dot(D)
    # print(idx_list)
    # print(A)
    swc_feat = swc_feat - swc_feat.min(axis=0)
    swc_feat_range = np.max(swc_feat, axis=0) - np.min(swc_feat, axis=0) + 1e-4
    # print(swc_feat_range.shape)
    swc_feat = swc_feat / swc_feat_range
    # print(swc_feat)
    return [swc_feat, A, idx_list]


def deal_data_for_remove(swc_path, is_start=False):
    Ma, Mp, SP_N, SP_step = 16, 16, 10, 1
    swc_name = os.path.basename(swc_path).replace('.swc', '')
    tif_path = swc_path.replace('.swc', '.tif')
    tif = read_tif(tif_path)

    swc_block = change_coord(swc_path)
    if is_start:
        swc_block[0, 1] = -1
    # print(tif_path)
    spe_feat = create_SPfeature(swc_block, tif, Ma, Mp, SP_N, SP_step)
    # swc_path = os.path.join(r'F:\Neruron_Repair\Neuron_Detect_Repair\data\data_for_detect\test\SP_feature', os.path.basename(swc_path).replace('.swc', ''))
    # spe_feat = loadspe(swc_path)
    spe_feat = spe_feat / spe_feat.max()
    subtree_feat = bulid_subgraphs(swc_block)
    data = []
    for subdata in subtree_feat:
        subswc_feat, subadj, idx_list = subdata
        subadj = torch.from_numpy(subadj).float()
        subswc_feat = torch.from_numpy(subswc_feat).float()
        subspe_feat = torch.from_numpy(spe_feat[idx_list]).float()
        # sublab = torch.from_numpy(np.eye(2)[swc_block[idx_list[0], 1].astype(np.int8)]).float()
        # if len(subspe_feat) == 19:
        #     print()
        # if len(subspe_feat) == 17:
        #     print('idx_list: ', idx_list)
        data.append([subadj, subswc_feat, subspe_feat])
    return swc_block, swc_name, data
