import os.path

from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers.utils.fx import torch_flip
import torch
import ipdb
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

with open('./feats/train_patch_features_with_names.pkl', 'rb') as f:
    train_patch_info = pickle.load(f)
with open('./feats/test_patch_features_with_names.pkl', 'rb') as f:
    test_patch_info = pickle.load(f)

sample_categories = np.array(train_patch_info['sample_categories'] + test_patch_info['sample_categories'])    # 0 - 171
train_categories = np.array(train_patch_info['sample_categories'])    # 0 - 171
test_categories = np.array(test_patch_info['sample_categories'])
train_len = train_categories.shape[0]
test_len = test_categories.shape[0]
# sample_ingre = np.array(patch_info['sample_ingre'])
# sample_paths = np.array(patch_info['sample_path'])
patch_names = np.array((train_patch_info['patch_names'] + test_patch_info['patch_names']))
train_patch_names = np.array(train_patch_info['patch_names'])
test_patch_names = np.array(test_patch_info['patch_names'])

image_names = np.array((train_patch_info['image_names'] + test_patch_info['image_names']))
train_image_names = np.array(train_patch_info['image_names'])
test_image_names = np.array(test_patch_info['image_names'])

# sample_category_ingre = np.char.add(np.char.add(image_names, '/'), patch_names)
sample_features = torch.cat((train_patch_info['sample_features'], test_patch_info['sample_features']), dim=0)
test_features = np.array(test_patch_info['sample_features'].cpu())
train_features = np.array(train_patch_info['sample_features'].cpu())


train_patch_word_relation = {}
for index, name in enumerate(train_patch_names):
    words = name.split('_')[-1][:-4]  # 'streaky pork slices'
    words_list = words.split(' ')
    train_patch_word_relation[index] = words_list
    
test_patch_word_relation = {}
for index, name in enumerate(test_patch_names):
    words = name.split('_')[-1][:-4]  # 'streaky pork slices'
    words_list = words.split(' ')
    test_patch_word_relation[index] = words_list


k = 2048
kmeans_name = f'./kmeans_ckpt/vireo_train_kmeans_{k}.pkl'
if os.path.exists(kmeans_name):
    with open(kmeans_name, 'rb') as f:
        kmeans = pickle.load(f)
else:
    kmeans = KMeans(n_clusters=k, random_state=0)
    # X = torch.load('./patch_feats_dino_thresh0.4.pth').cpu()
    X = train_features
    kmeans.fit(X)
    with open(kmeans_name, 'wb') as f:
        pickle.dump(kmeans, f)

labels = kmeans.labels_               # (n_sample,)
centroids = kmeans.cluster_centers_   # (K, 768)
cluster_counts = Counter(labels)
#
# top_clusters = [item[0] for item in cluster_counts.most_common(k)]
#
# category_cluster_centroids = {}
# category_confounder_centroids = {}
#
# # f = open(f'./{k}_all_cluster_info_by_all.txt', 'w')
# confusion_matrix = np.zeros((172, 172))

# train set kmeans ckpt
with open(kmeans_name, 'rb') as f:
    train_set_kmeans = pickle.load(f)
train_set_centers = train_set_kmeans.cluster_centers_
train_set_labels = train_set_kmeans.labels_


# this for loop is to statistic the word distribution of each cluster
cluster_word_dict = {}
cluster_patch_name_dict = {}
cluster_total_num_dict = {}
for cluster_id in range(k):
    word_count = defaultdict(int)  # 自动初始化为 0
    indices = np.where(train_set_labels == cluster_id)[0]
    cluster_total_num_dict[cluster_id] = (train_set_labels == cluster_id).sum()
    cluster_patch_name_dict[cluster_id] = train_patch_names[indices]
    
    for index in indices:
        for word in train_patch_word_relation[index]:
            word_count[word] += 1
            
    sorted_word_count = dict(sorted(word_count.items(), key=lambda x: x[1], reverse=True))
    cluster_word_dict[cluster_id] = dict(sorted_word_count)  # 转回普通字典
    


# print patch_names in clusters
# f = open('cluster_patch_names.txt', 'w')
# for cluster_id in range(k):
#     ids = np.where(cluster_id == train_set_labels)[0]
#     urls = []
#     for i in ids:
#         patch_url = sample_categories[i] + '/' + image_names[i] + '/' + patch_names[i]
#         urls.append(patch_url)
#     f.write(f'cluster id: {str(cluster_id):6}, {urls}\n')
# f.close()
# exit()


# f = open('cluster_words_info.txt', 'w')

centroids_info = {}
cnt = 0
for target_cluster_id in range(k):
    target_sample_id = train_set_labels == target_cluster_id   # bool
    target_sample_id = target_sample_id[:train_len]
    
    # target_category_ingre = sample_category_ingre[cluster_sample_id]
    # target_category_ingre_counts = Counter(target_category_ingre)
    target_category = train_categories[target_sample_id]
    # class-aware statistic
    target_category_counter = Counter(target_category)
    
    # word-level statistic
    dominant_words_list = []
    confusing_word_list = []
    cluster_words = cluster_word_dict[target_cluster_id]
    cluster_sample_num = target_sample_id.sum()
    for word, word_count in cluster_words.items():
        if word_count >= 0.5 * cluster_sample_num:
            dominant_words_list.append(word)
        else:
            confusing_word_list.append(word)
        
        
    # f.write(f'cluster id: {str(target_cluster_id):6}, num: {cluster_sample_num}')
    # f.write(str(dominant_words_list) + str(cluster_words) +'\n')

    # deal with cluster that contains no train samples
    # Actually, in vireo-food172, only 2 clusters contain no train samples
    if len(target_category_counter) == 0:
        distances = []
        for i in range(k):
            if i != target_cluster_id:
                d = np.linalg.norm(centroids[i] - centroids[target_cluster_id])
                distances.append((i, d))
        closest_cluster_id, _ = min(distances, key=lambda x: x[1])
        centroids_info[target_cluster_id] = centroids_info[closest_cluster_id]
        continue


    # pure cluster
    if len(target_category_counter) == 1:
        centroids_info[target_cluster_id] = {'dominant_category': target_category_counter.most_common(1)[0][0],
                                             'dominant_feature': centroids[target_cluster_id],
                                             'dominant_words': dominant_words_list,
                                             'confusing_words': confusing_word_list}
    else:
        top1_num = target_category_counter.most_common(1)[0][1]   # 最多的数值
        total_num = len(target_category)
        if top1_num > 0.9 * total_num:
            # dominant cluster
            confusing_category = [key for key, value in target_category_counter.items() if value < top1_num]
            centroids_info[target_cluster_id] = {'dominant_category': target_category_counter.most_common(1)[0][0],
                                                 'dominant_feature': centroids[target_cluster_id],
                                                 'confusing_category': confusing_category,
                                                 'dominant_words': dominant_words_list,
                                                 'confusing_words': confusing_word_list}
        else:
            # mix-cluster
            confusing_category = list(target_category_counter.keys())
            centroids_info[target_cluster_id] = {'confusing_category': confusing_category,
                                                 'confusing_feature': centroids[target_cluster_id],
                                                 'dominant_words': dominant_words_list,
                                                 'confusing_words': confusing_word_list}


train_info = {}
for i in range(len(train_patch_names)):
    patch_id = train_categories[i] + '/' + image_names[i] + '/' + train_patch_names[i]
    patch_centroid_id = labels[i]
    info = centroids_info[patch_centroid_id]
    if 'dominant_category' in info.keys():
        if info['dominant_category'] == sample_categories[i]:
            train_info[patch_id] = {'dominant_feature': info['dominant_feature'], 'dominant_words': info['dominant_words'], }
        else:
            train_info[patch_id] = {'confusing_feature': info['dominant_feature'], 'confusing_category': info['dominant_category'], 'dominant_words': info['dominant_words']}
    else:
        train_info[patch_id] = {'confusing_feature': info['confusing_feature'], 'confusing_category': info['confusing_category'], 'dominant_words': info['dominant_words']}





test_patch_info = {}
batch_size = 1000  # 每次处理 1000 个测试特征

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
    b_sq = np.sum(train_set_centers ** 2, axis=1).reshape(1, -1)  # (1, k)
    ab = np.dot(batch_features, train_set_centers.T)  # (batch_size, k)
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
        cluster_info_entry = train_info.get(closest_cluster_id, {})
        cluster_center = cluster_info_entry.get('cluster_center', None)
        confusing_category = cluster_info_entry.get('confusing_category', None)
        
        
        test_patch_info[patch_id] = {
            'cluster_center': cluster_center,
            'confusing_category': confusing_category
        }
    

'''
# # infer test sample info
# for index in range(test_len):
#     f = test_features[index]
#     distances = []
#     for i in range(k):
#         if True:
#             d = np.linalg.norm(f.cpu().numpy() - train_set_centers[i])
#             distances.append((i, d))
#     closest_cluster_id, _ = min(distances, key=lambda x: x[1])
#     closest_sample_id = train_set_labels == closest_cluster_id
#     closest_sample_categories = train_categories[closest_sample_id]
#     closest_sample_category_counter = Counter(closest_sample_categories)

#     # train_test_cluster_id = labels[index + train_len]
#     # train_test_sample_id = labels == train_test_cluster_id
#     # train_test_sample_categories = sample_categories[train_test_sample_id]
#     # train_test_sample_category_counter = Counter(train_test_sample_categories)
#     # print(train_test_sample_category_counter)
#     # ipdb.set_trace()


# ipdb.set_trace()
# sample_info = {}
# for i in range(len(patch_names)):
#     patch_id = sample_categories[i] + '/' + image_names[i] + '/' + patch_names[i]
#     patch_centroid_id = labels[i]
#     info = centroids_info[patch_centroid_id]
#     if 'dominant_category' in info.keys():
#         if info['dominant_category'] == sample_categories[i]:
#             sample_info[patch_id] = {'dominant_feature': info['dominant_feature'], 'semantic_words': info['semantic_words']}
#         else:
#             sample_info[patch_id] = {'confusing_feature': info['dominant_feature'], 'confusing_category': info['dominant_category'], 'semantic_words': info['semantic_words']}
#     else:
#         sample_info[patch_id] = {'confusing_feature': info['confusing_feature'], 'confusing_category': info['confusing_category'], 'semantic_words': info['semantic_words']}

# f = open('vireo_all_patch_info.pkl', 'wb')
# pickle.dump(sample_info, f)
# f.close()
# ipdb.set_trace()

    # if len(target_category_counts) >= 2:
    #     thresh = target_category_counts.most_common(1)[0][1] / 3  # 样本最多类别数量的1/3座作为阈值
    #     category_key = [int(it[0]) for it in target_category_counts.items() if it[1] > thresh]
    #     pairs = combinations(category_key, 2)
    #     for i, j in pairs:
    #         confusion_matrix[i, j] += 1
    #         confusion_matrix[j, i] += 1

if len(target_category_counter) < 2:
    category_key = target_category_counter.most_common(1)[0][0]
    if category_key not in category_cluster_centroids.keys():
        category_cluster_centroids[category_key] = []
    category_cluster_centroids[category_key].append(centroids[target_cluster_id])
else:
    thresh = target_category_counts.most_common(1)[0][1] / 3    # 样本最多类别数量的1/3座作为阈值
    category_key = [it[0] for it in target_category_counts.items() if it[1] > thresh]
    for k in category_key:
        if k not in category_confounder_centroids.keys():
            category_confounder_centroids[k] = []
        category_confounder_centroids[k].append(centroids[target_cluster_id])

# f.close()
# np.save('patch_derived_confusion_matrix.npy', confusion_matrix)
# ipdb.set_trace()

# prototype_tensor = torch.tensor([])
# confounder_tensor = torch.tensor([])
# for v in category_cluster_centroids.values():
#     v = torch.tensor(np.array(v))
#     prototype_tensor = torch.cat((prototype_tensor, v), 0)
# torch.save(prototype_tensor, 'prototype_tensor.pt')
# for v in category_confounder_centroids.values():
#     v = torch.tensor(np.array(v))
#     confounder_tensor = torch.cat((confounder_tensor, v), 0)
# torch.save(confounder_tensor, 'confounder_tensor.pt')
# category_prior = {
#     'prototype': category_cluster_centroids,
#     'confounder': category_confounder_centroids,
# }

# # labels = [i for i in range(0, 1000)]
# # counts = [item[1] for item in cluster_counts.most_common(1000)]
# # plt.figure(figsize=(20, 20))
# # plt.bar(labels, counts)
# # plt.xlabel('cluster label')
# # plt.ylabel('count')
# # plt.savefig('./freq.jpg')

# selected_clusters = [centroids[cluster] for cluster in top_clusters]

# # np.save(f'{k}_cluster_centers_', kmeans.cluster_centers_)
# # np.save(f'{k}_labels_', kmeans.labels_)
# np.save(f'./top_{nk}_centroids.npy', selected_clusters)
# print('done')
'''