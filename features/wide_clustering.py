import os.path
import json
from sklearn.cluster import MiniBatchKMeans, KMeans
import pickle
import ipdb
import torch
from collections import Counter

from sympy.integrals.meijerint_doc import category
from tqdm import tqdm
from sympy import centroid
import numpy as np

train_patch_info = torch.load('feats/wide_train_patch_features_with_names_thresh0.4&rank1.pt')

sample_features = train_patch_info['sample_features'].cpu()    # size = 177,646
sample_categories = train_patch_info['sample_categories'].cpu()
image_names = train_patch_info['image_names']
patch_names = train_patch_info['patch_names']

ipdb.set_trace()
K = 4096
stage = 'train'
kmeans_name = f'kmeans_ckpt/wide_{stage}_kmeans_{K}_results_kmeans.pkl'

if os.path.exists(kmeans_name):
    print('loading...')
    f = open(kmeans_name, 'rb')
    kmeans = pickle.load(f)
    f.close()
else:
    print('computing...')
    print('features length is:', sample_features.shape[0])
    kmeans = KMeans(n_clusters=K,
                    random_state=0)
    kmeans.fit(sample_features)
    print("Inertia:", kmeans.inertia_)
    w = open(kmeans_name, 'wb')
    pickle.dump(kmeans, w)
    w.close()

labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_
cluster_counts = Counter(labels)
top_clusters = [item[0] for item in cluster_counts.most_common(K)]

# w = open(f'wide_{K}_cluster_kmeans_info.txt', 'w')
cluster_info = {}
for target_cluster_id in top_clusters:
    dominant_category = []
    confusing_category = []
    target_sample_id = labels == target_cluster_id

    target_category = sample_categories[target_sample_id]
    cluster_sample_num = target_category.shape[0]
    target_category_sum = target_category.sum(0)
    value, indices = torch.sort(target_category_sum, descending=True)
    print('leading ratio:', value[0] / cluster_sample_num)

    dominant_thresh = cluster_sample_num * 0.5
    confusing_thresh = cluster_sample_num * 0.1

    for v, label in zip(value, indices):
        if v > dominant_thresh:
            dominant_category.append(label.item())
        elif v > confusing_thresh:
            confusing_category.append(label.item())
    cluster_info[int(target_cluster_id)] = {'dominant_category': dominant_category,
                                            'confusing_category': confusing_category,
                                            'cluster_center': cluster_centers[target_cluster_id], }


ipdb.set_trace()
# train set patch info
train_patch_info = {}
for i in range(sample_features.shape[0]):
    image_name = image_names[i].split('/',5)[-1]
    patch_id = f'{image_name}/{patch_names[i]}'
    patch_cluster_id = labels[i]
    info = cluster_info[patch_cluster_id]
    if info['confusing_category'] is not None:
        train_patch_info[patch_id] = info
print(f'length of confusing patches: {train_patch_info.__len__()} / {sample_features.shape[0]}')
torch.save(train_patch_info, f'patch_info/wide_train_patch_info_{K}.pt')

#####################################
# extract prototype-tensor
#####################################
# from collections import defaultdict
#
# f = open('/data_NUS_WIDE/NUS_labels/Concepts81.txt', 'r')
# idx2class = f.read().split('\n')[:-1]
# f.close()
#
# # w = open('wide_train_set_cluster_info.txt', 'w')
#
# category_cluster_center = defaultdict(list)
# for target_cluster_id in tqdm(top_clusters):
#     target_sample_id = labels == target_cluster_id
#     # patch_ids = [f'{image}/{patch}' for image, patch, mask in zip(image_names, patch_names, target_sample_id) if mask]
#     target_sample_category = sample_categories[target_sample_id]
#     cluster_sample_num = target_sample_category.shape[0]   # how many samples in target cluster
#     target_category_sum = target_sample_category.sum(0)    # (81, ) how many samples in each category
#     value, indices = torch.sort(target_category_sum, descending=True)
#     category_counter = [f'{idx2class[i]}: {v.int()}' for i, v in zip(indices, value) if v > 0]
#
#     # w.write(f'{target_cluster_id} - total {cluster_sample_num}: {category_counter}\n')
#
#     prototype_thresh = cluster_sample_num * 0.85   # 90% : result in 40 category with prototype   85% : 2494,768
#     for v, i in zip(value, indices):
#         category = i.item()
#         if v > prototype_thresh:
#             category_cluster_center[category].append(cluster_centers[target_cluster_id])
#
# # w.close()
# ipdb.set_trace()
# wide_prototype_tensor = torch.tensor([])
# for v in category_cluster_center.values():
#     v = torch.tensor(np.array(v))
#     wide_prototype_tensor = torch.cat((wide_prototype_tensor, v), 0)
#
# ipdb.set_trace()
# torch.save(wide_prototype_tensor, 'feats/wide_patch_prototype_thresh85.pt')
##########################################################################

# infer test patch info
ipdb.set_trace()
test_infos = torch.load('feats/wide_test_patch_features_with_names_thresh0.4&rank1.pt')
test_features = test_infos['sample_features'].cpu().numpy().astype(np.float32)  # 转为 NumPy 数组并使用 float32
test_categories = test_infos['sample_categories'].cpu().numpy()  # 假设为字符串或可转换为字符串
test_image_names = test_infos['image_names']
test_patch_names = test_infos['patch_names']

# 假设 cluster_centers 是一个 NumPy 数组，形状为 (k, feature_dim)
# 确保 cluster_centers 使用 float32
cluster_centers = cluster_centers.astype(np.float32)

# 假设 cluster_info 是一个字典，键为聚类 ID，值为包含 'cluster_center' 和 'confusing_category' 的字典
# 如果 cluster_info 是其他格式，请相应调整

# 初始化 test_patch_info 字典
test_patch_info = {}

# 设置批量大小，根据您的内存容量调整
batch_size = 1000  # 例如，每次处理 1000 个测试特征

num_test = test_features.shape[0]
num_batches = int(np.ceil(num_test / batch_size))

for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, num_test)
    batch_features = test_features[start_idx:end_idx]  # 形状 (batch_size, feature_dim)

    # 计算批量特征与所有质心之间的欧氏距离
    # 使用向量化的欧氏距离计算方法，避免生成 (batch_size, k, feature_dim) 的中间数组
    # 利用 (a - b)^2 = a^2 + b^2 - 2ab
    a_sq = np.sum(batch_features ** 2, axis=1).reshape(-1, 1)  # (batch_size, 1)
    b_sq = np.sum(cluster_centers ** 2, axis=1).reshape(1, -1)  # (1, k)
    ab = np.dot(batch_features, cluster_centers.T)  # (batch_size, k)
    distances = np.sqrt(a_sq + b_sq - 2 * ab)  # (batch_size, k)

    # 找到每个测试特征最近的聚类 ID
    closest_cluster_ids = np.argmin(distances, axis=1)  # (batch_size,)

    for i in range(start_idx, end_idx):
        local_idx = i - start_idx  # 当前批次内的索引
        # 构建 patch_id，假设 test_categories 是字符串类型或可以转换为字符串
        image_name = test_image_names[i].split('/',5)[-1]
        patch_id = f'{image_name}/{test_patch_names[i]}'

        closest_cluster_id = closest_cluster_ids[local_idx]

        # 获取对应聚类的 'cluster_center' 和 'confusing_category'
        cluster_info_entry = cluster_info.get(closest_cluster_id, {})
        cluster_center = cluster_info_entry.get('cluster_center', None)
        confusing_category = cluster_info_entry.get('confusing_category', None)

        test_patch_info[patch_id] = {
            'cluster_center': cluster_center,
            'confusing_category': confusing_category
        }

print(f'test confusing patch num: {test_patch_info.__len__()} / {test_features.shape[0]}')
ipdb.set_trace()
torch.save(test_patch_info, f'patch_info/wide_test_patch_info_{K}.pt')


