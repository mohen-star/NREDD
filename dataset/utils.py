import math
import os
import numpy as np
import torch
from libtiff import TIFF
from scipy.ndimage import map_coordinates
from scipy.spatial.distance import pdist, squareform, cdist
from datetime import datetime
import torch.nn.functional as F


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
            print('%d %d %.3f %.3f %.3f %.3f %d' %
                  tuple(swc[i, :].tolist()), file=f)


def loadtif(tif_path):
    images = []
    tif = TIFF.open(tif_path, mode='r')
    for image in tif.iter_images():
        # image = np.flipud(image)
        images.append(image)
    return np.array(images)


def savetif(img, path):
    # if type == 'img':
    tif = TIFF.open(path, mode='w')
    num = img.shape[0]
    for i in range(num):
        # img[i] = np.flipud(img[i])
        tif.write_image(((img[i]).astype(np.uint8)), compression=None)
    tif.close()
    return


def savetif_for_vaa3dshow(img, path):
    # if type == 'img':
    tif = TIFF.open(path, mode='w')
    num = img.shape[0]
    for i in range(num):
        img[i] = np.flipud(img[i])
        tif.write_image(((img[i]).astype(np.uint8)), compression=None)
    tif.close()
    return


def find_child_idx(swc, idx0):  # 寻找当前节点的所有子节点
    IDX = []  # 节点的标号
    for i in range(swc.shape[0]):
        if swc[i, 6] == idx0:
            IDX.append(i)
    return IDX


def find_parent_idx(swc, idx6):
    IDX = []
    for i in range(swc.shape[0]):
        if swc[i, 0] == idx6:
            IDX.append(i)
    return IDX


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
    return swc[:, 0:7]


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
    # dis_edge = compute_adjacency_matrix_with_angle(swc, sigma=8.0)
    # swc_adj = np.maximum(swc_adj, dis_edge)
    D = np.power(np.sum(swc_adj, axis=-1), -0.5)
    D[np.isinf(D)] = 0
    D = np.diag(D)
    A = swc_adj.dot(D).transpose().dot(D)
    return A


def get_edge_for_coonect_test(swc, angle_weight=True):
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
    # if angle_weight:
    #     dis_edge = compute_adjacency_matrix_with_angle(swc, sigma=8.0)
    #     swc_adj = np.maximum(swc_adj, dis_edge)
    return swc_adj


def get_edge_for_remove(swc):
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
    # adj_org = swc_adj
    dis_edge = compute_adjacency_matrix(swc, sigma=8.0)
    # print(dis_edge[0])
    swc_adj = np.maximum(swc_adj, dis_edge)
    return swc_adj


def compute_adjacency_matrix(swc_block, num_neighbors=10, sigma=1.0):
    # compute adj with distance around points
    distances = pdist(swc_block[:, 2:5], metric='euclidean')
    dist_list = squareform(distances)
    edge = np.zeros((swc_block.shape[0], swc_block.shape[0]), dtype=np.float32)
    near_idx = np.argpartition(dist_list, num_neighbors, axis=1)[:, :num_neighbors]
    for i in range(near_idx.shape[0]):
        dists = dist_list[i, near_idx[i]]
        weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
        weights = weights / np.sum(weights)
        # edge[i, near_idx[i]] = weights
        edge[i, near_idx[i]] = np.where(weights >= 0.1, 1, 0)
    return edge


def find_link_change(point1, point2, swc):
    # 1. 构建字典，分别存储每个节点的父节点和子节点
    parent_dict = {}
    child_dict = {}

    # 遍历 SWC，填充字典
    for node in swc:
        idx, _, _, _, _, parent_idx = node[0], node[1], node[2], node[3], node[4], node[6]
        if parent_idx != -1:
            if parent_idx not in child_dict:
                child_dict[parent_idx] = []
            child_dict[parent_idx].append(idx)

        parent_dict[idx] = parent_idx

    # 2. 向上遍历，找到所有父节点
    parent_idx_list = set()  # 用 set 来加速查找
    parent_idx = point1
    while parent_idx != -1:
        if parent_idx in parent_idx_list:
            break  # 遇到已访问的节点，避免循环
        parent_idx_list.add(parent_idx)
        parent_idx = parent_dict.get(parent_idx, -1)  # 获取父节点，若无父节点返回 -1

    # 3. 向下遍历，找到所有子节点
    child_idx_list = set()  # 用 set 来加速查找
    child_idx_refresh = [point1]
    while child_idx_refresh:
        new_child_idx_refresh = []
        for idx in child_idx_refresh:
            if idx in child_idx_list:
                continue  # 已访问过的节点不再处理
            child_idx_list.add(idx)
            if idx in child_dict:
                new_child_idx_refresh.extend(child_dict[idx])
        child_idx_refresh = new_child_idx_refresh

    # 4. 检查 point2 是否在父节点或子节点列表中
    if point2 in parent_idx_list or point2 in child_idx_list:
        return True

    return False


def compute_adjacency_matrix_with_angle(swc_block, num_neighbors=10, sigma=8.0, sigma_angle=20.0):
    """
    计算神经元节点的邻接矩阵，考虑距离和角度的影响，角度越大，权重越大。

    参数：
    swc_block：SWC 块（numpy 数组，包含每个节点的坐标和结构信息）
    num_neighbors：每个节点的邻居数量（默认10个）
    sigma：高斯函数的距离标准差（控制距离对权重的影响）
    sigma_angle：控制角度对权重的影响（越小表示角度差异越敏感）

    返回：
    edge：邻接矩阵，表示节点间的权重
    """
    # 计算成对的欧几里得距离
    distances = cdist(swc_block[:, 2:5], swc_block[:, 2:5], metric='euclidean')

    # 初始化邻接矩阵
    edge = np.zeros((swc_block.shape[0], swc_block.shape[0]), dtype=np.float32)

    # 找出每个点最近的 num_neighbors 个邻居的索引
    near_idx = np.argpartition(distances, num_neighbors, axis=1)[:, :num_neighbors]

    for i in range(near_idx.shape[0]):
        # 初始化需要删除的邻居的索引
        det_idx = []

        # 筛选满足条件的邻居
        for j in range(len(near_idx[i])):
            neighbor = near_idx[i][j]
            if find_link_change(i + 1, neighbor + 1, swc_block):
                det_idx.append(j)

        # 使用布尔索引删除不满足条件的邻居
        near_idx_temp = np.array([idx for idx in near_idx[i] if idx not in [near_idx[i][k] for k in det_idx]])

        if len(near_idx_temp) > 0:
            # 计算角度并筛选符合角度条件的邻居
            angles = compute_angle(i, near_idx_temp, swc_block)
            valid_idx = np.where(angles >= 100)[0]  # 筛选出角度大于等于100度的邻居
            near_idx_temp = near_idx_temp[valid_idx]
            angles = angles[valid_idx]

            # 确保有满足角度条件的邻居
            if len(near_idx_temp) > 0:
                # 计算这些邻居的距离
                dists = distances[i, near_idx_temp]

                # 计算距离的高斯权重
                distance_weights = np.exp(-dists ** 2 / (2 * sigma ** 2))

                # 计算角度权重，角度越大，权重越大
                angle_weights = np.exp(angles / sigma_angle)

                # 综合距离权重和角度权重
                final_weights = distance_weights * angle_weights

                # 归一化权重，使其和为1
                final_weights = final_weights / np.sum(final_weights)
                # edge[i, near_idx_temp] = final_weights
                edge[i, near_idx_temp] = np.where(final_weights >= 0.2, 1, 0)
            # else:
            #     print(f"Warning: No valid neighbors found for node {i} with angle filter.")
    return edge


def compute_angle(start_idx, end_idxs, swc):
    """
    计算给定神经元节点之间的夹角，夹角由起始点、父节点和多个结束点的坐标决定。

    参数：
    start_idx：起始点的索引
    end_idxs：结束点的索引列表
    swc：SWC 格式的坐标数组

    返回：
    angles：包含所有结束点与起始点夹角的数组（单位：度）
    """
    # 获取父节点索引，首先尝试通过 start_idx 的父节点索引来查找
    start_land_idx = np.where(swc[:, 0] == swc[start_idx, 6])[0]

    # 如果没有找到父节点，则尝试查找子节点的父节点
    if len(start_land_idx) == 0:
        start_land_idx = np.where(swc[:, 6] == swc[start_idx, 0])[0]
        if len(start_land_idx) == 0:
            # print(f"警告: 无法找到起始点 {start_idx} 的父节点或连接点，返回默认角度 90°")
            return np.full(len(end_idxs), 90.0)  # 返回一个包含 90° 的数组

    # 获取起始点和连接点的坐标
    start_point_xyz = swc[start_idx, 2:5]
    start_land_point_xyz = swc[start_land_idx[0], 2:5]  # 可能会有多个匹配，取第一个

    # 计算每个结束点的夹角
    angles = []
    for end_idx in end_idxs:
        end_point_xyz = swc[end_idx, 2:5]
        end_land_idx = np.where(swc[:, 0] == swc[end_idx, 6])[0]
        if len(end_land_idx) == 0:
            end_land_idx = np.where(swc[:, 6] == swc[end_idx, 0])[0]
            if len(end_land_idx) == 0:
                # print(f"警告: 无法找到起始点 {end_idx} 的父节点或连接点，返回默认角度 90°")
                angle = 90.0
            else:
                end_land_point_xyz = swc[end_land_idx[0], 2:5]
                angle = calculate_angle_AC(start_point_xyz, start_land_point_xyz, end_point_xyz, end_land_point_xyz)
        else:
            end_land_point_xyz = swc[end_land_idx[0], 2:5]
            angle = calculate_angle_AC(start_point_xyz, start_land_point_xyz, end_point_xyz, end_land_point_xyz)
        angles.append(angle)
    return np.array(angles)


def calculate_angle_AC(A, B, C, D):
    """
    计算由四个点 A、B、C、D 定义的夹角，夹角由向量 AB 和 CD 之间的夹角决定。
    其中 A 和 C 是夹角的顶点。
    参数：
    A, B, C, D：三维空间中的四个点坐标（numpy 数组或列表）
    返回：
    夹角（单位：度）。如果输入无效，返回 NaN。
    """
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)

    # 计算向量 AB 和 CD
    AB = B - A
    CD = D - C

    # 计算向量的点积
    dot_product = np.dot(AB, CD)

    # 计算向量 AB 和 CD 的模长
    magnitude_AB = np.linalg.norm(AB)
    magnitude_CD = np.linalg.norm(CD)

    # 防止模长为零导致除零错误
    if magnitude_AB == 0 or magnitude_CD == 0:
        print("无效输入，至少一个向量的长度为零：")
        print(f"A: {A}, B: {B}, C: {C}, D: {D}")
        return np.nan

        # 计算夹角的余弦值
    cos_theta = dot_product / (magnitude_AB * magnitude_CD)

    # 限制余弦值在 -1 到 1 之间，以防浮动误差
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # 计算夹角（弧度制）
    theta_rad = np.arccos(cos_theta)

    # 将夹角转换为角度
    theta_deg = np.degrees(theta_rad)

    return theta_deg


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


def create_SPfeature(swc, tif, swc_SP_root, Ma=16, Mp=16, SP_N=10, SP_step=1):
    # Ma = Ma
    # Mp = Mp
    # SP_N = 10
    # SP_step = 1
    SP_core = generate_sphere(Ma, Mp)
    img_normalized = tif / 255.0
    gamma = 1.0
    img_gamma = np.power(img_normalized, gamma)
    img = np.uint8(img_gamma * 255)
    tif = img

    for i in range(swc.shape[0]):
        xyz = swc[i, 2:5]
        yxz = xyz[[1, 0, 2]]
        # print(yxz)
        Spherical_patch = Spherical_Patches_Extraction(tif, yxz, SP_N, SP_core, SP_step)
        SP = Spherical_patch.reshape([Ma, Mp, SP_N - 1]).transpose([2, 0, 1])
        # SP_list.append(SP)
        sp_save_path = os.path.join(swc_SP_root, str(i) + '_.tif')
        savetif(SP, sp_save_path)
    return True


def feature_normal_image(feature_tif):
    # pmax = feature_tif.max()
    # if pmax > 0:
    #     feature_tif = feature_tif / pmax
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


def rotate_points_around_point(points, angle, axis, center):
    # Define rotation matrices
    Rx = lambda theta: np.array([[1, 0, 0],
                                 [0, np.cos(theta), -np.sin(theta)],
                                 [0, np.sin(theta), np.cos(theta)]])

    Ry = lambda theta: np.array([[np.cos(theta), 0, np.sin(theta)],
                                 [0, 1, 0],
                                 [-np.sin(theta), 0, np.cos(theta)]])

    Rz = lambda theta: np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])

    angle = np.radians(angle)
    if axis == 'x':
        R = Rx(angle)
    elif axis == 'y':
        R = Ry(angle)
    elif axis == 'z':
        R = Rz(angle)
    else:
        raise ValueError("The axis must be 'x', 'y', or 'z'")

    # Translate points to origin (subtract the center)
    translated_points = points - center.reshape(3, 1)

    # Rotate the points
    rotated_points = R.dot(translated_points)

    # Translate points back (add the center)
    rotated_points += center.reshape(3, 1)

    return rotated_points


def rotate_3d_image(image, angle, axis, center=None):
    if center is None:
        center = np.array(image.shape) // 2  # Default to the center of the image

    center = np.array(center)  # Ensure center is a numpy array

    # Create meshgrid of image coordinates (x, y, z)
    x, y, z = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), np.arange(image.shape[2]),
                          indexing='ij')

    # Translate image to origin by subtracting the center
    x_translated = x - center[0]
    y_translated = y - center[1]
    z_translated = z - center[2]

    # Convert angle to radians
    angle_rad = np.radians(angle)

    # Apply the rotation matrix based on the specified axis
    if axis == 'z':
        y_rot = y_translated * np.cos(angle_rad) - z_translated * np.sin(angle_rad)
        z_rot = y_translated * np.sin(angle_rad) + z_translated * np.cos(angle_rad)
        x_rot = x_translated  # X-coordinate stays the same
    elif axis == 'y':
        # Rotation around the Y axis
        x_rot = x_translated * np.cos(angle_rad) + z_translated * np.sin(angle_rad)
        z_rot = -x_translated * np.sin(angle_rad) + z_translated * np.cos(angle_rad)
        y_rot = y_translated  # Y-coordinate stays the same
    elif axis == 'x':
        # Rotation around the Z axis
        x_rot = x_translated * np.cos(angle_rad) - y_translated * np.sin(angle_rad)
        y_rot = x_translated * np.sin(angle_rad) + y_translated * np.cos(angle_rad)
        z_rot = z_translated  # Z-coordinate stays the same
    else:
        raise ValueError("Invalid axis. Please choose from 'x', 'y', or 'z'.")

    # Translate back the rotated image to the original center
    x_rot += center[0]
    y_rot += center[1]
    z_rot += center[2]

    coords = np.array([x_rot.flatten(), y_rot.flatten(), z_rot.flatten()])
    rotated_image = map_coordinates(image, coords, order=1, mode='nearest')
    rotated_image = rotated_image.reshape(image.shape)
    return rotated_image


def data_augmentation_rotate(swc, image, angles, axes):
    swc_roated = []
    tif_roated = []
    points = swc[:, 2:5].T
    center = np.array([128, 128, 128])  # Center point around which to rotate
    for axis in axes:
        for angle in angles:
            rotated_points = rotate_points_around_point(points, angle, axis, center)
            rotated_tif = rotate_3d_image(image, angle, axis, center)
            swc_copy = swc.copy()
            swc_copy[:, 2:5] = rotated_points.T
            # swc_copy = swc_copy[np.random.permutation(swc_copy.shape[0])]
            # swc_copy = resample_swc_(swc_copy)
            swc_roated.append(swc_copy)
            tif_roated.append(rotated_tif)
    return swc_roated, tif_roated


def load_data_for_remove(root, is_augmentation=False, Ma=16, Mp=16):
    path_list = []
    adj_list = []
    feature_xyz_list = []
    label_list = []
    block_root = os.path.join(root, 'labeled_data')
    for swc_name in os.listdir(block_root):
        data_path = os.path.join(block_root, swc_name)
        label_path = data_path.replace('labeled_data', 'label').replace('.swc', '.txt')
        image_path = data_path.replace('labeled_data', 'image').replace('.swc', '.tif')
        SP_root = data_path.replace('labeled_data', 'SP_feature').replace('.swc', '')

        data = loadswc(data_path)
        image = loadtif(image_path)
        # label = np.loadtxt(label_path)
        adj = get_edge_for_remove(data)
        if not os.path.exists(SP_root):
            os.makedirs(SP_root)
            create_SPfeature(data, image, SP_root, Ma=Ma, Mp=Mp)

        adj_list.append(adj)
        # label_list.append(label)
        feature_xyz_list.append(norm_feature_xyz(data[:, 2:5]))
        path_list.append(data_path)
        print(data_path)

        if is_augmentation:
            angles = [90, 180, 270]
            axes = ['z']  # z轴向分辨率低，对于球面特征提取，保持z轴不动，其特征图像会于原始图像提取较为相似，从而保证增广图像对于原始图像的真实性
            if not os.path.exists(os.path.join(root, 'data_aug')):
                os.makedirs(os.path.join(root, 'data_aug'))
            if not os.path.exists(
                    data_path.replace('labeled_data', 'data_aug').replace('.swc', '.' + axes[0] + '90.swc')):
                data_rotates, image_rotates = data_augmentation_rotate(data, image, angles, axes)
                for i in range(len(data_rotates)):
                    axis = axes[i // 3]
                    angle = str(angles[i % 3])
                    rotate_data = data_rotates[i]
                    rotate_image = image_rotates[i]
                    data_save_path = data_path.replace('labeled_data', 'data_aug').replace('.swc',
                                                                                           '.' + axis + angle + '.swc')
                    image_save_path = image_path.replace('image', 'data_aug').replace('.tif',
                                                                                      '.' + axis + angle + '.tif')
                    saveswc(data_save_path, rotate_data)
                    savetif(rotate_image, image_save_path)
            for i in range(len(angles) * len(axes)):
                axis = axes[i // 3]
                angle = str(angles[i % 3])
                rotate_data_path = data_path.replace('labeled_data', 'data_aug').replace('.swc',
                                                                                         '.' + axis + angle + '.swc')
                rotate_image_path = image_path.replace('image', 'data_aug').replace('.tif', '.' + axis + angle + '.tif')
                rotate_data = loadswc(rotate_data_path)
                rotate_image = loadtif(rotate_image_path)
                SP_aug_root = SP_root + '.' + axis + angle
                if not os.path.exists(SP_aug_root):
                    os.makedirs(SP_aug_root)
                    create_SPfeature(rotate_data, rotate_image, SP_aug_root)

                feature_xyz_list.append(norm_feature_xyz(rotate_data[:, 2:5]))
                adj = get_edge_for_remove(rotate_data)
                adj_list.append(adj)
                # label_list.append(label)
                path_list.append(rotate_data_path)
    return feature_xyz_list, adj_list, label_list, path_list


def load_data_for_connect(root, Ma, Mp):
    path_list = []
    adj_list = []
    feature_xyz_list = []
    label_list = []
    block_root = os.path.join(root, 'raw_data')
    for swc_name in os.listdir(block_root):
        data_path = os.path.join(block_root, swc_name)
        gt_path = data_path.replace('raw_data', 'gt')
        label_path = data_path.replace('raw_data', 'label').replace('.swc', '.txt')
        image_path = data_path.replace('raw_data', 'image').replace('.swc', '.tif')
        SP_root = data_path.replace('raw_data', 'SP_feature').replace('.swc', '')

        data = loadswc(data_path)
        image = loadtif(image_path)
        adj = get_edge_for_coonect(data)
        if os.path.exists(label_path):
            label = np.loadtxt(label_path)
        else:
            gt = loadswc(gt_path)
            label = get_label(data, gt)
            if not os.path.exists(os.path.dirname(label_path)):
                os.makedirs(os.path.dirname(label_path))
            np.savetxt(label_path, label, delimiter=' ')

        if not os.path.exists(SP_root):
            os.makedirs(SP_root)
            create_SPfeature(data, image, SP_root, Ma=Ma, Mp=Mp)

        feature_xyz_list.append(norm_feature_xyz(data[:, 2:5]))
        adj_list.append(adj)
        label_list.append(label)
        path_list.append(data_path)
        # print(data_path)
    return feature_xyz_list, adj_list, label_list, path_list


def get_label(swc, swc_gt):
    label_adj = np.zeros((swc.shape[0], swc.shape[0])).astype(np.float32)
    for i in range(swc_gt.shape[0]):
        cur_point = swc_gt[i, 0].astype(np.int32)
        par_point = swc_gt[i, -1].astype(np.int32)
        dist = np.linalg.norm(swc[:, 2:5] - swc_gt[i, 2:5], axis=1)
        current_idx = np.where(dist < 0.5)
        swc_cur_idx = np.where(np.linalg.norm(swc[:, 2:5] - swc_gt[cur_point - 1, 2:5], axis=1) < 0.5)
        swc_par_idx = np.where(np.linalg.norm(swc[:, 2:5] - swc_gt[par_point - 1, 2:5], axis=1) < 0.5)
        # print(swc_cur_idx, swc_par_idx)

        if par_point == -1:
            label_adj[swc_cur_idx, swc_cur_idx] = 1
        else:
            label_adj[swc_cur_idx, swc_cur_idx] = 1
            label_adj[swc_cur_idx, swc_par_idx] = 1

        cld_list = np.where(swc_gt[:, -1] == cur_point)[0]
        for cld_idx in cld_list:
            cld_point = swc_gt[cld_idx, 0].astype(np.int32)
            swc_cld_idx = np.where(np.linalg.norm(swc[:, 2:5] - swc_gt[cld_point - 1, 2:5], axis=1) < 0.5)
            label_adj[swc_cur_idx, swc_cld_idx] = 1
        # if swc_cur_idx[0] == 308:
        #     print(cur_point)
        #     print(par_point)
        #     print(cld_list)
    return label_adj


def load_SPfeature(swc_path):
    if 'data_aug' in swc_path:
        sp_root = swc_path.replace('data_aug', 'SP_feature').replace('.swc', '')
    else:
        sp_root = swc_path.replace('raw_data', 'SP_feature').replace('.swc', '')
    sp_feature_list = []
    for i in range(len(os.listdir(sp_root))):
        sp_path = os.path.join(sp_root, str(i) + '_.tif')
        # print(sp_path)
        sp_feature = loadtif(sp_path)
        sp_feature_list.append(sp_feature)

    feat_sp = np.array(sp_feature_list).astype(np.float32)
    feat_sp = feature_normal_image(feat_sp)
    feat_sp = torch.from_numpy(feat_sp).float()
    return feat_sp


def load_feature_for_SPE(swc_path):
    if 'data_aug' in swc_path:
        sp_root = swc_path.replace('data_aug', 'SP_feature').replace('.swc', '')
    else:
        sp_root = swc_path.replace('raw_data', 'SP_feature').replace('.swc', '')
    sp_feature_list = []
    for file_name in os.listdir(sp_root):
        sp_path = os.path.join(sp_root, file_name)
        sp_feature = loadtif(sp_path)
        sp_feature = np.array(sp_feature).astype(np.float32)
        sp_feature = feature_normal_image(sp_feature)
        sp_feature = torch.from_numpy(sp_feature).float()
        sp_feature = torch.unsqueeze(sp_feature, 1)
        sp_feature_list.append(sp_feature)

    return sp_feature_list


class Node(object):
    def __init__(self, idx, position, radius, node_type=3):
        self.position = position
        self.radius = radius
        self.node_type = node_type
        self.nbr = []
        self.idx = idx


def generate_Ntree(swc):
    Node_list = []
    for i in range(0, swc.shape[0]):
        node = Node(idx=i, position=swc[i, 2:5], radius=swc[i, 5], node_type=swc[i, 1])
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
    swc_feat = swc_feat - np.min(swc_feat, axis=0)
    swc_feat_range = np.max(swc_feat, axis=0) - np.min(swc_feat, axis=0) + 1e-4
    # print(swc_feat_range.shape)
    swc_feat = swc_feat / swc_feat_range
    # print(swc_feat)
    return [swc_feat, A, idx_list]


def bulid_subgraphs(swc, deeplayer=4):
    Tree = generate_Ntree(swc)
    subtree_list = []
    for i in range(len(Tree)):
        cur_stree = []
        cur_node_list = []
        next_node_list = []
        cur_node_list.append(i)
        # print("start node: ",cur_node_list)
        for j in range(deeplayer):
            while (len(cur_node_list) > 0):
                cur_node = cur_node_list.pop()
                cur_stree.append(cur_node)
                # print(cur_node)
                for nbr_node_id in Tree[cur_node].nbr:
                    if nbr_node_id not in cur_stree:
                        # print("nbr: ",nbr_node_id)
                        next_node_list.append(nbr_node_id)
            cur_node_list = next_node_list.copy()
            next_node_list.clear()
        cur_stree1 = [Tree[idx] for idx in cur_stree]
        subtree_list.append(bulid_swc_feat(cur_stree1))
    return subtree_list

def loadspe(path):
    point_list = os.listdir(path)
    # spe_feat = np.zeros((len(point_list),9,64,64))
    spe_feat = np.zeros((len(point_list), 9, 16, 16))
    for point_num in point_list:
        # print(point_num)
        point_spe = loadtif(os.path.join(path, point_num))
        # print(point_spe.shape)
        # point_spe = point_spe.reshape(1,1024)
        cur_index = int(point_num[:-5])
        # print(cur_index)
        spe_feat[cur_index] = point_spe
    return spe_feat

if __name__ == '__main__':
    # data_root = r'E:\Neure\18454_reconstruction\2'
    # feature_xyz_list, adj_list, label_list, path_list = load_data(data_root, mode='train')
    data_root = r'F:\Neruron_Repair\Neuron_Detect_Repair\data\data_for_detect\train'
    feature_xyz_list, adj_list, label_list, path_list = load_data_for_remove(data_root, is_augmentation=True)



