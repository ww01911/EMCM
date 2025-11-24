import io
import traceback
import random

import pickle
from unittest.mock import patch

import scipy.io as matio
import numpy as np
from PIL import Image
import torch
import torch.utils.data
import os
import torchvision.transforms as transforms
from sympy.series.limitseq import dominant

from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
# import cv2
import ipdb
import random
import time
import re
from tqdm import tqdm
import json

PAD, CLS = '[PAD]', '[CLS]'
MASK = '[MASK]'


def default_loader(image_path):
    return Image.open(image_path).convert('RGB')


def padding_tensor(t, padding_len=10):
    if len(t.shape) == 1:
        t = t.unsqueeze(0)
    length = min(t.size(0), padding_len)
    padded_tensor = torch.zeros((padding_len, *t.size()[1:]), dtype=t.dtype, device=t.device)
    padded_tensor[:length] = t[:length]

    mask = torch.ones(padding_len)
    mask[:length] = 0.0
    return padded_tensor, mask, length


transform_ = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def build_dataset(stage, tokenizer, args):
    if args.dataset == 'vireo':
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        ds = Vireo172(args.img_path, args.f_path, patch_path=args.patch_path, stage=stage, tokenizer=tokenizer,
                      transform=transform, device=args.device, pred_text_path=args.pred_text_path,
                      patch_pad_len=args.patch_pad_len, text_pad_len=args.text_pad_len, topk=args.topk,
                      extra_image_info_path=args.extra_image_info_path, load_dino_info=args.load_dino_info,
                      confounder_path=args.confounder_path, patch_level_info=args.patch_level_info,
                      patch_level_info_k=args.patch_level_info_k, thresh=args.thresh)
            
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(ds)
            test_sampler = torch.utils.data.distributed.DistributedSampler(ds)
        else:
            train_sampler = None
            test_sampler = None

        if stage == 'train':
            return DataLoader(dataset=ds, batch_size=args.batch_size,
                              shuffle=train_sampler is None, num_workers=args.worker, pin_memory=True,
                              sampler=train_sampler)
        elif stage == 'test':
            return DataLoader(dataset=ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.worker, pin_memory=True, sampler=test_sampler)

    elif args.dataset == 'wide':
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        ds = NusWide(args.img_path, args.f_path, stage, tokenizer, transform=transform, patch_root=args.patch_path,
                     patch_pad_len=args.patch_pad_len, patch_info_path=args.patch_info_path, background_root=None,
                     pred_tag_path=args.pred_tag_path, patch_level_info_k=args.patch_level_info_k)
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(ds)
            test_sampler = torch.utils.data.distributed.DistributedSampler(ds)
        else:
            train_sampler = None
            test_sampler = None
        if stage == 'train':
            return DataLoader(ds, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.worker, pin_memory=True,
                              sampler=train_sampler)
        else:
            return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.worker, pin_memory=False,
                              sampler=test_sampler)


def get_confounder(args):
    confounder = Confounder(info_path='/mnt/dino/ingre2image_info/sorted_filtered_word2label2image_info.json')
    return DataLoader(confounder, batch_size=args.batch_size, shuffle=False, num_workers=args.worker, pin_memory=True)


def get_patch(args, root):
    patch_set = PatchLoader(patch_root=root)
    return DataLoader(patch_set, batch_size=args.batch_size, shuffle=False, num_workers=args.worker, pin_memory=True)


class NusWide(Dataset):
    def __init__(self, img_path, f_path, stage, tokenizer, pad_size=32, transform=None, patch_root=None,
                 patch_pad_len=None, patch_info_path=False, background_root=None, pred_tag_path=None, patch_level_info_k=4096):
        self.img_path = img_path
        self.f_path = f_path  # file path
        self.tokenizer = tokenizer
        self.pad_size = pad_size
        self.stage = stage
        if stage == 'train':
            list_image = 'TR'
            label_path = self.f_path + '/NUS_train_labels'
        elif stage == 'test':
            list_image = 'TE'
            label_path = self.f_path + '/NUS_test_labels'
        else:
            list_image = 'VAL'
            label_path = ""

        img_path_file = os.path.join(f_path, list_image)
        with io.open(img_path_file, encoding='utf-8') as file:
            img_paths = file.read().split('\n')[:-1]
        self.img_paths = img_paths

        with io.open(label_path, encoding='utf-8') as file:
            path_to_label = file.read().split('\n')[:-1]

        self.img_label = []
        for path in path_to_label:
            label_str = path.split(' ')
            label_str.pop()
            self.img_label.append([float(i) for i in label_str])
        self.img_label = torch.tensor(self.img_label)

        # text
        with open('/data_NUS_WIDE/TagList1k.txt', 'r') as f:
            idx2words = f.read().split('\n')
            idx2words = idx2words[:1000]
        if stage == 'train':
            indexVectors = np.load(self.f_path + '/np_train_word_vector.npy')
        elif stage == 'test':
            indexVectors = np.load(self.f_path + '/np_test_word_vector.npy')
        else:
            indexVectors = None
        self.idx2words = idx2words
        self.indexVectors = indexVectors.astype(int)
        self.transform = transform
        self.loader = default_loader

        self.bg_root = background_root

        if patch_root is not None:
            self.patch_path = patch_root + f'/wide/{stage.upper()}_REGIONS_BY_PREDICTING_TOP5'
            self.patch_pad_len = patch_pad_len
            if patch_info_path:
                self.patch_info = torch.load(f'{patch_info_path}/wide_{stage}_patch_info_{patch_level_info_k}.pt')
            else:
                self.patch_info = None
        else:
            self.patch_path = None
            self.patch_info = None

        if pred_tag_path:
            self.pred_tag = torch.load(pred_tag_path + f'/wide_{stage}_top5_tags.pth', map_location='cpu')
        else:
            self.pred_tag = None

    @staticmethod
    def multi_label_one_hot(num_class, label_indices_list):
        one_hot = torch.zeros(1, num_class)
        for indices in label_indices_list:
            one_hot[0, indices - 1] = 1
        return one_hot.view(-1)

    def __getitem__(self, index):
        # load image
        f_name = self.img_paths[index]
        image = self.loader(self.img_path + f_name)
        if self.transform is not None:
            image = self.transform(image)
        label = self.img_label[index]

        if self.bg_root:
            bg_image = self.loader(self.bg_root + f_name)
            if self.transform is not None:
                bg_image = self.transform(bg_image)

        # load text & padding
        indexVector = self.indexVectors[index, :]
        nonzero_idx = indexVector[indexVector != 0]
        tag_label = self.multi_label_one_hot(num_class=1000, label_indices_list=nonzero_idx)
        words_list = [self.idx2words[i - 1] for i in nonzero_idx]
        content = ' '.join(words_list)
        token = self.tokenizer.tokenize(content)
        token = [CLS] + token
        seq_len = len(token)
        text_tokens = self.tokenizer.convert_tokens_to_ids(token)
        if len(token) < self.pad_size:
            mask = [1] * len(text_tokens) + [0] * (self.pad_size - len(token))
            text_tokens += ([0] * (self.pad_size - len(token)))
        else:
            mask = [1] * self.pad_size
            text_tokens = text_tokens[:self.pad_size]
            seq_len = self.pad_size
        text_tokens = torch.tensor([text_tokens]).squeeze(0)
        mask = torch.tensor([mask]).squeeze(0)

        if self.pred_tag is not None:
            pred_tag = self.pred_tag[index]
            pred_tag_words = [self.idx2words[i] for i in pred_tag]
            tag_str = ' '.join(pred_tag_words)
            pred_tokens = self.tokenizer.tokenize(tag_str)
            pred_tokens = self.tokenizer.convert_tokens_to_ids(pred_tokens)
            if len(pred_tokens) < self.pad_size:
                pred_ingre_mask = [1] * len(pred_tokens) + [0] * (self.pad_size - len(pred_tokens))
                pred_tokens += ([0] * (self.pad_size - len(pred_tokens)))
            else:
                pred_ingre_mask = [1] * self.pad_size
                pred_tokens = pred_tokens[:self.pad_size]
            pred_tokens = torch.tensor([pred_tokens]).squeeze(0)
            pred_mask = torch.tensor([pred_ingre_mask]).squeeze(0)


        # load patches & padding
        if self.patch_path is not None:
            patch_path = self.patch_path + f_name
            patch_names = os.listdir(patch_path)

            patches_list = []
            for patch in patch_names:
                if not patch.endswith('.jpg'):
                    patch_names.remove(patch)
                    continue
                # filter confidence
                confidence = float(patch.split('_')[1])
                if confidence < 0.15:
                    patch_names.remove(patch)
                    continue

                p_ = self.loader(os.path.join(patch_path, patch))
                if self.transform is not None:
                    p_ = self.transform(p_)
                patches_list.append(p_)
            if not patches_list:
                patch_names.append(f_name)
                patches_list.append(image)
            patches = torch.stack(patches_list)
            patches, patch_mask, patch_num = padding_tensor(patches, self.patch_pad_len)

        # ipdb.set_trace()
        if self.patch_info is not None:
            confusing_label = label.detach().clone()
            confusing_features = []
            prototype_features = []
            for name in patch_names:
                patch_id = f'{f_name[1:]}/{name}'
                if patch_id in self.patch_info.keys():
                    info = self.patch_info[patch_id]
                    # dominant_category = info['dominant_category']
                    confusing_category = info['confusing_category']
                    if confusing_category:
                        confusing_features.append(info['cluster_center'])
                        for category in confusing_category:
                            confusing_label[category] = 1
            confusing_label = confusing_label - label
            confusing_label = confusing_label.detach()
            assert torch.all(confusing_label >= 0), 'improper label range.'
            confusing_label = torch.clamp(confusing_label, 0, 1)
            # for category in dominant_category:
            #     if label[category] == 1:
            #         prototype_features.append(info['cluster_center'])

            if not confusing_features:
                confusing_features.append(torch.zeros(768))
            confusing_features = torch.from_numpy(np.array(confusing_features)).float()
            confusing_features, confusing_mask, _ = padding_tensor(confusing_features, 5)

        rt_dict = {
            'id': index,
            'f_name': f_name,
            'image': image,
            'label': label,
            'tag_label': tag_label,
        }

        if self.bg_root:
            rt_dict.update({
                'bg_image': bg_image,
            })

        if self.stage == 'train':
            rt_dict.update({
                'text_tokens': text_tokens,
                'text_masks': mask,
            })
        if self.pred_tag is not None:
            rt_dict.update({
                'pred_text_tokens': pred_tokens,
                'pred_text_mask': pred_mask,
            })
        if self.patch_path is not None:
            rt_dict.update({
                'patches': patches,
                'patch_mask': patch_mask,
                'patch_num': patch_num,
                # 'patch_name': patch_names,
            })
        if self.patch_info is not None:
            rt_dict.update({
                'confusing_features': confusing_features,
                'confusing_features_mask': confusing_mask,
                'confusing_label': confusing_label,
            })
        if self.pred_tag is not None:
            rt_dict.update({

            })
        # ipdb.set_trace()
        return rt_dict

    def __len__(self):
        return len(self.img_paths)


class Vireo172(torch.utils.data.Dataset):
    def __init__(self, img_path, text_path, patch_path=None, stage='train', tokenizer=None, pad_size=32,
                 image_loader=default_loader, transform=None, device=None, pred_text_path=None,
                 patch_pad_len=20, text_pad_len=25, topk=3, extra_image_info_path=None, load_dino_info=False,
                 confounder_path=None, patch_level_info=False, patch_level_info_k=2048, thresh=0.1):
        self.stage = stage
        self.device = device
        self.pad_size = pad_size
        self.dino_text_pad_size = 100
        self.image_path = img_path + '/ready_chinese_food'
        self.text_path = os.path.join(text_path, 'SplitAndIngreLabel')
        self.patch_pad_len = patch_pad_len
        self.extra_pad_len = 2 * patch_pad_len
        self.text_pad_len = text_pad_len
        self.load_graph = False
        self.thresh = thresh   # to filter low_quality patches

        # self.match_words = False
        self.typical_feat_dict = torch.load('dict/typical_feat_dict.pt', map_location='cpu')
        self.avg_category_typical_feat = torch.load('dict/category_avg_typical_feat.pt', map_location='cpu')

        with open('dict/ingre_category_info.pkl', 'rb') as f:
            self.ingre_category_info = pickle.load(f)
        self.ingredient_words_relation = np.load('ingredient_words_relation.npy')  # (312, 312)
        with open('IngredientWordsLowerCaseSorted.txt', encoding='utf-8') as file:
            self.words2idx = {word.strip(): idx for idx, word in enumerate(file)}

        self.confusion_matrix = np.load('features/patch_derived_confusion_matrix.npy')

        self.words2idx['double'] = self.words2idx['side'] = 91
        self.words2idx['tea'] = 284
        self.words2idx['broad'] = 32

        if stage == 'train':
            list_image = 'TR'
            self.patch_path = (patch_path + f'/TRAIN_REGIONS_BY_PREDICTING_TOP{topk}_THRESH') if patch_path else None
            # self.patch_path = (patch_path + f'/vireo/TRAIN_REGIONS_BY_PREDICTING_TOP{topk}_0.1') if patch_path else None
            pred_text_path = (pred_text_path + f'/train_ingre_predict_top{topk}.pth') if pred_text_path else None
            self.all_patch_feats = torch.load('vireo_train_sample_patch_feats.pt', map_location='cpu')
            self.all_patch_feats = None
        elif stage == 'test':
            list_image = 'TE'
            self.patch_path = (patch_path + f'/TEST_REGIONS_BY_PREDICTING_TOP{topk}_THRESH') if patch_path else None
            # self.patch_path = (patch_path + f'/vireo/TEST_REGIONS_BY_PREDICTING_TOP{topk}_0.1') if patch_path else None
            pred_text_path = (pred_text_path + f'/test_ingre_predict_top{topk}.pth') if pred_text_path else None
            self.all_patch_feats = torch.load('vireo_test_sample_patch_feats.pt', map_location='cpu')
            self.all_patch_feats = None
        else:
            list_image = 'VAL'

        self.ingre2image_path = patch_path + f'/TRAIN_REGIONS_BY_PREDICTING_TOP{topk}_THRESH' if patch_path else None

        # patch data
        img_path_file = os.path.join(self.text_path, list_image + '.txt')
        # lmdb_path = patch_path + f'/{stage}_regions_by_predicting_top3.lmdb'
        # self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
        # pred ingredient data
        if load_dino_info:
            pred_info_path = patch_path + f"/regions_{stage}_by_predicting_top{topk}_thresh.txt"
            with open(pred_info_path, encoding='utf-8') as file:
                self.info_strs = file.read().split('\n')
        else:
            self.info_strs = None

        # patch-level confusing/dominant info
        if patch_level_info:
            # with open(patch_level_info_path, 'rb') as file:
            #     self.patch_level_info = pickle.load(file)
            self.patch_level_info = torch.load(f'features/patch_info/vireo_{stage}_patch_info_pure_and_confused_{patch_level_info_k}.pt')
        else:
            self.patch_level_info = None

        # extra patch data
        if extra_image_info_path:
            with open(extra_image_info_path, encoding='utf-8') as f:
                self.extra_image_info = json.load(f)
        else:
            self.extra_image_info = None

        with io.open(img_path_file, encoding='utf-8') as file:
            self.path_to_images = file.read().split('\n')

        indexVectors = matio.loadmat(os.path.join(self.text_path, f'indexVector_{stage}.mat'))[f'indexVector_{stage}']
        self.indexVectors = indexVectors.astype(np.compat.long)

        self.pred_text_vector = torch.load(pred_text_path, map_location='cpu') if pred_text_path else None

        if confounder_path:
            self.confounder_path = confounder_path
            # dict {word: tensor (768)}
            self.confounder_dict = torch.load(confounder_path, map_location='cpu')
        else:
            self.confounder_dict = None

        ingredient_list_path = os.path.join(self.text_path, 'IngredientList.txt')
        with io.open(ingredient_list_path, encoding='utf-8') as f:
            self.index2words = f.read().split('\n')

        self.tokenizer = tokenizer
        self.image_loader = image_loader
        self.transform = transform
        # self.pattern = re.compile(r'^\d*_\D*.jpg$')
        self.num_workers = 10

    @staticmethod
    def multi_label_one_hot(num_class, label_indices_list):
        one_hot = torch.zeros(1, num_class)
        for indices in label_indices_list:
            one_hot[0, indices - 1] = 1
        return one_hot.view(-1)

    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def _load_images(self, paths):
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            images = list(executor.map(self._load_image, paths))
        return images

    def __getitem__(self, index):
        # load gt text info
        # ipdb.set_trace()
        idx = self.indexVectors[index, :]
        non_zero_idx = idx[idx != 0]
        ingre_labels = self.multi_label_one_hot(num_class=353, label_indices_list=non_zero_idx)
        words_list = [self.index2words[i - 1] for i in non_zero_idx]
        content = ' '.join(words_list)
        content_words = set(content.lower().split())
        token = self.tokenizer.tokenize(content)
        token = [CLS] + token
        seq_len = len(token)
        text_tokens = self.tokenizer.convert_tokens_to_ids(token)
        if len(text_tokens) < self.pad_size:
            mask = [1] * len(text_tokens) + [0] * (self.pad_size - len(token))
            text_tokens += ([0] * (self.pad_size - len(token)))
        else:
            mask = [1] * self.pad_size
            text_tokens = text_tokens[:self.pad_size]
        # use list to pass a copy of text_tokens
        text_tokens = torch.tensor([text_tokens]).squeeze(0)
        mask = torch.tensor([mask]).squeeze(0)

        # if needed pred ingre info
        if self.pred_text_vector is not None:
            pred_idx = self.pred_text_vector[index, :]
            pred_none_zero = pred_idx[pred_idx != 0].to(torch.long)
            pred_ingre = self.multi_label_one_hot(num_class=353, label_indices_list=pred_none_zero)
            pred_ingre_list = [self.index2words[i - 1] for i in pred_none_zero]
            pred_ingre_str = ' '.join(pred_ingre_list)
            pred_ingre_tokens = self.tokenizer.tokenize(pred_ingre_str)
            
            pred_ingre_tokens = self.tokenizer.convert_tokens_to_ids(pred_ingre_tokens)
            if len(pred_ingre_tokens) < self.pad_size:
                pred_ingre_mask = [1] * len(pred_ingre_tokens) + [0] * (self.pad_size - len(pred_ingre_tokens))
                pred_ingre_tokens += ([0] * (self.pad_size - len(pred_ingre_tokens)))
            else:
                pred_ingre_mask = [1] * self.pad_size
                pred_ingre_tokens = pred_ingre_tokens[:self.pad_size]
            pred_ingre_tokens = torch.tensor([pred_ingre_tokens]).squeeze(0)
            pred_ingre_mask = torch.tensor([pred_ingre_mask]).squeeze(0)

        # get label
        path = self.path_to_images[index]
        f_name = path.split('/')[2]
        image = self.image_loader(self.image_path + path)
        ori_image = np.array(image, dtype=np.uint8)
        if self.transform is not None:
            image = self.transform(image)
        label_str = path.split('/')[1]
        label = int(label_str) - 1
        label = torch.tensor([label])

        # load confusion category probability distribution
        prob = self.confusion_matrix[label]   # (172, 172)
        prob[label] = 0
        prob = prob / (prob.sum() + 1e-8)
        if prob.sum() == 0:
            prob = np.ones_like(prob) / prob.shape[0]

        # load text tags of each patch and concat together
        if self.info_strs is not None:
            info_str = self.info_strs[index]
            list_part = '[' + info_str.split(' [')[1].split('],')[0] + ']'
            list_elements = eval(list_part)

            dino_text_split = [len(e.split(' ')) for e in list_elements]  # words length list

            dino_str = ' '.join(list_elements)
            dino_text_tokens = self.tokenizer.tokenize(dino_str)
            if not dino_text_tokens:
                dino_text_tokens.append('nothing')
            dino_text_tokens = self.tokenizer.convert_tokens_to_ids(dino_text_tokens)

            if len(dino_text_tokens) < self.dino_text_pad_size:
                dino_text_masks = [1] * len(dino_text_tokens) + [0] * (self.dino_text_pad_size - len(dino_text_tokens))
                dino_text_tokens += ([0] * (self.dino_text_pad_size - len(dino_text_tokens)))
            else:
                dino_text_masks = [1] * self.dino_text_pad_size
                dino_text_tokens = dino_text_tokens[:self.dino_text_pad_size]

            if len(dino_text_split) < self.patch_pad_len:
                dino_text_split += ([0] * (self.patch_pad_len - len(dino_text_split)))
            else:
                dino_text_split = dino_text_split[:self.patch_pad_len]
            dino_text_split = torch.tensor([dino_text_split]).squeeze(0)
            dino_text_tokens = torch.tensor([dino_text_tokens]).squeeze(0)
            dino_text_masks = torch.tensor([dino_text_masks]).squeeze(0)
            
            
        load_confounding_feat_not_filter_patch = True  # to decide (1) filter out confounding patch  or (2) generate confusing features

        # ipdb.set_trace()
        if self.patch_path:       
            if self.all_patch_feats is not None:
                patch_feats = self.all_patch_feats[index]
            # info_str = self.info_strs[index]
            # list_part = '[' + info_str.split(' [')[1].split('],')[0] + ']'
            # list_elements = eval(list_part)

            # if self.match_words:
            #     for i, name in enumerate(list_elements, start=1):
            #         patch_name = name.lower().split()
            #         patch_name_token = self.tokenizer.tokenize(patch_name)
            #         patch_name_token = self.tokenizer.convert_tokens_to_ids(patch_name_token)

            # region_names = [f"{i}_{name}.jpg" for i, name in enumerate(list_elements, start=1)]
            region_root = os.path.join(self.patch_path, str(label.item()), f_name)
            region_names = os.listdir(region_root)
            # only patch in /vireo need label+1 !!   2025-1-20
            patch_ids = [os.path.join(str(label.item()), f_name, r_name) for r_name in region_names if r_name.endswith('.jpg') and r_name != f_name]
            patches_list = []
            patch_names = []
            
            # get dino-derived text infos
            dino_pad_len = 3
            dino_tokens_all = []
            dino_tokens_mask_all = []
            for r_name in region_names:
                if r_name != f_name:
                    dino_text = r_name.split('_')[-1][:-4]
                    dino_tokens = self.tokenizer.tokenize(dino_text)
                    dino_tokens = self.tokenizer.convert_tokens_to_ids(dino_tokens)
                    if len(dino_tokens) < dino_pad_len:
                        dino_tokens_mask = [1] * len(dino_tokens) + [0] * (dino_pad_len - len(dino_tokens))
                        dino_tokens += ([0] * (dino_pad_len - len(dino_tokens)))
                    else:
                        dino_tokens_mask = [1] * dino_pad_len
                        dino_tokens = dino_tokens[:dino_pad_len]
                    dino_tokens = torch.tensor([dino_tokens]).squeeze(0)
                    dino_tokens_mask = torch.tensor([dino_tokens_mask]).squeeze(0)
                    dino_tokens_all.append(dino_tokens)
                    dino_tokens_mask_all.append(dino_tokens_mask)
            if len(dino_tokens_all) == 0:
                dino_tokens_all = pred_ingre_tokens[:dino_pad_len]
                dino_tokens_mask_all = pred_ingre_mask[:dino_pad_len]
            else:
                try:
                    dino_tokens_all = torch.stack(dino_tokens_all)          
                    dino_tokens_mask_all = torch.stack(dino_tokens_mask_all)
                except RuntimeError:
                    print(region_names, region_root)
                    exit(0)
            # padding
            dino_tokens_all, _, _  = padding_tensor(dino_tokens_all, self.patch_pad_len)
            dino_tokens_mask_all, _, _ = padding_tensor(dino_tokens_mask_all, self.patch_pad_len)
                    
            
            # ipdb.set_trace()
            if not load_confounding_feat_not_filter_patch:
                # only load non-confounding patches
                for patch_id in patch_ids:
                    if patch_id not in self.patch_level_info.keys():
                        r_ = self.image_loader(os.path.join(self.patch_path, patch_id))
                        if self.transform is not None:
                            r_ = self.transform(r_)
                        patches_list.append(r_)
                        patch_names.append(patch_id.split('/')[-1])

            else:
                # load all patches, but subtract confounding-feat later
                for r_image in region_names:
                    if r_image == f_name or not r_image.endswith('.jpg'):
                        continue
                    confidence = float(r_image.split('_')[1])
                    if confidence < self.thresh:
                        continue

                    r_ = self.image_loader(os.path.join(region_root, r_image))
                    if self.transform is not None:
                        r_ = self.transform(r_)
                    patches_list.append(r_)
                    patch_names.append(r_image)

            if not patches_list:
                patch_names.append(f_name)
                patches_list.append(image)

            assert len(patches_list) == len(patch_names), 'unequal length'

            # typical_list = []
            # rare_list = []
            # # label is unavailable during testing
            #
            # typical_words = self.typical_feat_dict[label.item()].keys()
            # for r_image in region_names:
            #     if r_image == f_name:
            #         continue
            #
            #     r_ = self.image_loader(os.path.join(region_root, r_image))
            #     if self.transform is not None:
            #         r_ = self.transform(r_)
            #     if any(w in r_image for w in typical_words):
            #         typical_list.append(r_)
            #     else:
            #         rare_list.append(r_)
            # rare_num = len(rare_list)
            # if rare_num == 0:
            #     split_point = len(typical_list) // 2
            #     if split_point == 0:
            #         split_point = 1
            #     rare_list = typical_list[:split_point]
            #     if not rare_list:
            #         rare_list.append(image)
            #     typical_list = typical_list[split_point:]
            #     rare_num = len(rare_list)
            # patches_list = rare_list + typical_list

            # rare_mask = torch.ones(self.patch_pad_len)
            # rare_mask[:rare_num] = 0.0

            # patches = self._load_images(region_names)

            # region_path = [f"{label}/{f_name}/{i}_{name}.jpg" for i, name in enumerate(list_elements, start=1)]
            # patches = []
            # start = time.time()
            #
            # with self.env.begin(write=False) as txn:
            #     for k in region_path:
            #         value = txn.get(k.encode())
            #         image_array = np.frombuffer(value, dtype=np.uint8)
            #         r_ = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            #         if self.transform:
            #             r_ = transform_(r_)
            #         patches.append(r_)
            # print(f"lmdb: {time.time() - start}")
            
            
            '''if load_confounding_feat_not_filter_patch:
                if self.patch_level_info is not None:
                    prototype_features = []
                    confusing_features = []
                    confusing_words = []
                    confusing_categories = set()
                    try:
                        for patch_id in patch_ids:
                            if patch_id in self.patch_level_info.keys():
                                patch_info = self.patch_level_info[patch_id]
                                if patch_info['confusing_category'] is not None:
                                # if 'dominant_feature' in patch_info.keys():
                                #     if 'dominant_category' not in patch_info.keys() or patch_info[
                                #         'dominant_category'] == str(label.item()):
                                #         prototype_features.append(patch_info['dominant_feature'])
                                #     else:
                                #         confusing_features.append(patch_info['confusing_feature'])
                                #         confusing_categories.update(patch_info['confusing_category'])
                                # else:
                                #     confusing_features.append(patch_info['confusing_feature'])
                                #     confusing_categories.update(patch_info['confusing_category'])
                                    confusing_features.append(patch_info['cluster_center'])
                                    confusing_categories.update(patch_info['confusing_category'])
                                    # confusing_words += patch_info['dominant_words']
                    except KeyError as e:
                        print(e)
                        ipdb.set_trace()

                    confusing_categories.discard(str(label.item()))
                    if not prototype_features:
                        prototype_features.append(np.zeros(768))
                    if not confusing_features:
                        confusing_features.append(np.zeros(768))
                    prototype_features = torch.tensor(np.array(prototype_features)).float()
                    confusing_features = torch.tensor(np.array(confusing_features)).float()
                    prototype_features, prototype_features_mask, _ = padding_tensor(prototype_features, 5)
                    confusing_features, confusing_features_mask, _ = padding_tensor(confusing_features, 5)
                    confusing_label = torch.zeros(172)
                    for c in confusing_categories:
                        confusing_label[int(c)] = 1
                    if not torch.any(confusing_label):
                        confusing_label = torch.ones(172)
                        confusing_label[int(label)] = 0
            '''
            
            proto_features = []
            confusing_features = []
            confusing_words = []
            confusing_categories = set()
            if load_confounding_feat_not_filter_patch:
                is_confused = torch.zeros(self.patch_pad_len)
                if self.patch_level_info is not None:
                    for idx, patch_id in enumerate(patch_ids):
                        if not patch_id in self.patch_level_info.keys():
                            proto_features.append(torch.zeros(768))
                            continue
                        patch_info = self.patch_level_info[patch_id]
                        proto_features.append(patch_info['cluster_center'])
                        if patch_info['is_confused']:
                            is_confused[idx] = 1
                            confusing_features.append(patch_info['cluster_center'])
                            confusing_categories.update(patch_info['confusing_category'])
                            
                    confusing_categories.discard(str(label.item()))
                    if not confusing_features:
                        confusing_features.append(np.zeros(768))
                    confusing_features = torch.tensor(np.array(confusing_features)).float()
                    proto_features = torch.tensor(np.array(proto_features)).float()
                    confusing_features, confusing_features_mask, _ = padding_tensor(confusing_features, 10)
                    proto_features, proto_mask, _ = padding_tensor(proto_features, 10)
                    confusing_label = torch.zeros(172)
                    for c in confusing_categories:
                        confusing_label[int(c)] = 1
                    if not torch.any(confusing_label):
                        confusing_label = torch.ones(172)
                        confusing_label[int(label)] = 0
                        
                    
            patch_num = min(len(patches_list), self.patch_pad_len)
            patches = torch.stack(patches_list)
            c, h, w = patches[0].shape
            zero_vectors = torch.zeros(self.patch_pad_len, c, h, w)
            zero_vectors[:patch_num, :, :, :] = patches[:patch_num, :, :, :]
            patches = zero_vectors

            patch_mask = torch.ones(self.patch_pad_len)
            patch_mask[:patch_num] = 0.0
            
            
            # filtering pred ingre tokens
            # ipdb.set_trace()
            pred_ingre_str = ' '.join(pred_ingre_list)
            pred_ingre_tokens = self.tokenizer.tokenize(pred_ingre_str)
            pred_ingre_tokens = [x for x in pred_ingre_tokens if x not in confusing_words]
            pred_ingre_tokens = self.tokenizer.convert_tokens_to_ids(pred_ingre_tokens)
            if len(pred_ingre_tokens) < self.pad_size:
                pred_ingre_mask = [1] * len(pred_ingre_tokens) + [0] * (self.pad_size - len(pred_ingre_tokens))
                pred_ingre_tokens += ([0] * (self.pad_size - len(pred_ingre_tokens)))
            else:
                pred_ingre_mask = [1] * self.pad_size
                pred_ingre_tokens = pred_ingre_tokens[:self.pad_size]
            pred_ingre_tokens = torch.tensor([pred_ingre_tokens]).squeeze(0)
            pred_ingre_mask = torch.tensor([pred_ingre_mask]).squeeze(0)

            if self.extra_image_info:
                extra_patches = patches_list
                if self.stage == 'train':
                    pred_words = pred_ingre_str.lower().split(' ')
                    extra_list = []
                    for word in pred_words:
                        try:
                            # load extra patch from other images with same category and same ingre word
                            w_extra_images = self.extra_image_info[word][str(label.item())]
                        except KeyError:
                            continue
                        x = random.choice(w_extra_images)
                        extra_list.append(os.path.join(self.ingre2image_path, x["path"]))

                    for x in extra_list:
                        x_ = self.image_loader(x)
                        if self.transform is not None:
                            x_ = self.transform(x_)
                        extra_patches.append(x_)

                # 实验结果表明测试时加入随机类别的 patch 会严重掉点
                # elif self.stage == 'test':
                #     pred_words = pred_ingre_str.lower().split(' ')
                #     extra_list = []
                #     for word in pred_words:
                #         try:
                #             # load extra patch from other images from any possible category but same word
                #             chosen_label = random.choice(list(self.extra_image_info[word].keys()))
                #             w_extra_images = self.extra_image_info[word][chosen_label]
                #         except KeyError:
                #             continue
                #         x = random.choice(w_extra_images)
                #         extra_list.append(os.path.join(self.ingre2image_path, x["path"]))
                #
                #     for x in extra_list:
                #         x_ = self.image_loader(x)
                #         if self.transform is not None:
                #             x_ = self.transform(x_)
                #         extra_patches.append(x_)

                extra_patch_num = min(len(extra_patches), self.extra_pad_len)
                extra_patches = torch.stack(extra_patches)
                zero_vectors = torch.zeros(self.extra_pad_len, c, h, w)
                zero_vectors[:extra_patch_num, :, :, :] = extra_patches[:extra_patch_num, :, :, :]
                extra_patches = zero_vectors
                extra_patch_mask = torch.ones(self.extra_pad_len)
                extra_patch_mask[:extra_patch_num] = 0.0

            region_name_str = ",".join(region_names)
            patch_name_str = ",".join(patch_names)

            if self.confounder_dict is not None:
                # ipdb.set_trace()
                pred_words = pred_ingre_str.lower().split(' ')
                confounder_list = []
                for word in pred_words:
                    try:
                        confounder_list.append(self.confounder_dict[word])
                    except KeyError:
                        continue
                confounder = torch.stack(confounder_list)

                # for word in pred_words:
                #     if word not in self.confounder_dict:
                #         continue
                #     ingre_category_dict = self.confounder_dict[word]
                #     ingre_confounder = torch.stack([x for x in ingre_category_dict.values()])
                #     confounder_list.append(ingre_confounder)
                # confounder = torch.cat(confounder_list, dim=0).contiguous()

                confounder, confounder_mask, confounder_num = padding_tensor(confounder, self.patch_pad_len)

            # load graph
            if self.load_graph:
                graph, relation = self.build_graph(
                    info_str=self.info_strs[index])  # relation of regions and words (from dino)
                pad_width = ((0, pred_ingre_tokens.shape[0] - relation.shape[0]),
                             (0, pred_ingre_tokens.shape[0] - relation.shape[0]))
                relation = np.pad(relation, pad_width=pad_width, mode='constant', constant_values=0)
            else:
                graph = 0
                relation = 0

        rt_dict = {
            'sample_id': index,
            'f_name': path,
            'image': image,
            'ori_image': ori_image,
            'label': label,

        }
        if self.pred_text_vector is not None:
            rt_dict.update({
                'pred_text_tokens': pred_ingre_tokens,
                'pred_text_mask': pred_ingre_mask,
            })

        if self.stage == "train":
            # only use gt text infos when training
            rt_dict.update({
                'text_tokens': text_tokens,
                'text_masks': mask,
            })
            if hasattr(self, 'confusion_matrix'):
                rt_dict.update({
                    'confusion_prob': prob,
                })
        if self.info_strs is not None:
            rt_dict.update({
                'dino_text_tokens': dino_text_tokens,
                'dino_text_masks': dino_text_masks,
                'dino_text_split': dino_text_split,
                'region_name_str': region_name_str,
            })
        if self.patch_path:
            rt_dict.update({
                'patches': patches,
                'patch_num': patch_num,
                'patch_mask': patch_mask,
                'patch_name_str': patch_name_str
                # 'rare_mask': rare_mask,
            })
            rt_dict.update({
                'dino_tokens_all': dino_tokens_all,
                'dino_tokens_mask_all': dino_tokens_mask_all,
            })
            rt_dict.update({
                'is_confused': is_confused,
                'proto_features': proto_features,
            })
        if self.all_patch_feats is not None:
            rt_dict.update({
                'patch_feats': patch_feats,
            })
        if self.patch_level_info and load_confounding_feat_not_filter_patch:
            rt_dict.update({
                'confusing_label': confusing_label,
                # 'prototype_features': prototype_features,
                # 'prototype_features_mask': prototype_features_mask,
                'confusing_features': confusing_features,
                'confusing_features_mask': confusing_features_mask,
            })
        if self.extra_image_info:
            rt_dict.update({
                'extra_patches': extra_patches,
                'extra_patch_num': extra_patch_num,
                'extra_patch_mask': extra_patch_mask,
            })
        if self.confounder_dict is not None:
            rt_dict.update({
                'confounder': confounder,
                'confounder_mask': confounder_mask,
                'confounder_num': confounder_num,
            })
        if self.avg_category_typical_feat is not None:
            rt_dict.update({
                'avg_category_typical_feat': self.avg_category_typical_feat,
            })
        return rt_dict

    def __len__(self):
        return len(self.indexVectors)

    def build_graph(self, info_str):
        matrix_size = self.patch_pad_len + self.text_pad_len
        graph = np.eye(matrix_size)
        list_part = '[' + info_str.split(' [')[1].split('],')[0] + ']'
        words_part = info_str.split('], ')[1]
        list_elements = eval(list_part)
        words = re.findall(r'\b\w+\b', words_part.lower())
        if self.ingredient_words_relation is not None:
            indices = [self.words2idx[w] for w in words]
            l = len(indices)
            relation = self.ingredient_words_relation[np.ix_(indices, indices)]
            # normalized_relation = (relation - np.min(relation)) / (np.max(relation) - np.min(relation))
            # graph[self.patch_pad_len: self.patch_pad_len + l, self.patch_pad_len: self.patch_pad_len + l] = (
            #     normalized_relation + np.eye(l))
        for i, element in enumerate(list_elements):
            element_words = re.findall(r'\b\w+\b', element)
            for w in element_words:
                j = (words.index(w)) if w in words else None
                if j:
                    graph[i, self.patch_pad_len + j] = 1
                    graph[self.patch_pad_len + j, i] = 1
        D = np.diag(np.sum(graph, axis=1))

        try:
            D_inv_sqrt = np.linalg.inv(np.sqrt(D))
        except np.linalg.LinAlgError:
            print(np.diag(D))
        graph_normalized = D_inv_sqrt @ graph @ D_inv_sqrt
        return graph_normalized, relation


def create_graph_in_batch(batch_data, tfidf_matrix, matrix_word2idx, tokenizer):
    image, label, token_ids, content = batch_data
    label = label.view(-1)
    bs = image.shape[0]
    # key: bert_ids -> value: ordered index
    batch_tokens_dict = {}
    # ordered index -> words
    idx2words = {}
    idx = 0
    for token_id, ingre in zip(token_ids, content):
        token_id = token_id[token_id.nonzero()].squeeze(1).tolist()
        words = tokenizer.convert_ids_to_tokens(token_id)
        if len(token_id) != len(words):
            print('not equal')
        for t, w in zip(token_id, words):
            t = int(t)
            if t not in batch_tokens_dict:
                batch_tokens_dict[t] = idx
                idx2words[idx] = w
                idx = idx + 1
    batch_tokens_count = idx
    graph = np.eye(batch_tokens_count + bs)
    for i in range(bs):
        token = token_ids[i]
        token = token[token.nonzero()]
        for ids in token:
            ids = int(ids)
            try:
                graph[i][batch_tokens_dict[ids] + bs] = tfidf_matrix[label[i]][
                    matrix_word2idx[idx2words[batch_tokens_dict[ids]]]]
            except KeyError:
                traceback.print_exc()
                import re
                cur_key = idx2words[batch_tokens_dict[ids]]
                if cur_key == '-':
                    continue
                cur_key = rf'(\b|-){cur_key}(\b|-)'
                fuzzy_matched_keys = [key for key in matrix_word2idx if re.search(cur_key, key)]
                fuzzy_matched_keys = fuzzy_matched_keys[0]
                graph[i][batch_tokens_dict[ids] + bs] = tfidf_matrix[label[i]][matrix_word2idx[fuzzy_matched_keys]]
                continue
    # symmetric adjacency matrix and normalize
    graph = np.maximum(graph, graph.T)
    D = np.diag(np.sum(graph, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    graph_normalized = D_inv_sqrt @ graph @ D_inv_sqrt

    token_ids = list(batch_tokens_dict.keys())
    token_ids = torch.tensor(token_ids).unsqueeze(0)
    mask = torch.ones_like(token_ids)

    return image, (token_ids, mask), label, graph_normalized


class Confounder(Dataset):
    def __init__(self, info_path, image_loader=default_loader):
        super().__init__()
        with open(info_path, 'r') as f:
            self.ingre_patch_info = json.load(f)
        self.patch_paths = []
        self.patch_category = []
        self.patch_ingre = []
        self.root = '/mnt/dino/TRAIN_REGIONS_BY_PREDICTING_TOP3_THRESH/'
        self.image_loader = image_loader
        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # ipdb.set_trace()
        for word, labels in self.ingre_patch_info.items():
            for label, value in labels.items():
                # if len(value) < 50:
                #     continue
                for path in value:
                    if eval(path['logit']) <= 0.4:  # filter    0.4 -> patch num = 105,460
                        continue
                    self.patch_paths.append(path['path'])
                    self.patch_category.append(label)
                    self.patch_ingre.append(word)
        # ipdb.set_trace()
        assert len(self.patch_paths) == len(self.patch_category) == len(self.patch_ingre), "unequal length"

    def __getitem__(self, index):
        # ipdb.set_trace()
        patch_path = self.root + self.patch_paths[index]
        patch_category = self.patch_category[index]
        patch_ingre = self.patch_ingre[index]
        patch = self.image_loader(patch_path)
        if self.transform is not None:
            patch = self.transform(patch)

        return patch, patch_category, patch_ingre

    def __len__(self):
        return len(self.patch_paths)


class PatchLoader(Dataset):
    # use this class to load all patches
    def __init__(self, patch_root, image_loader=default_loader):
        self.stage = "train" if "TRAIN" in patch_root else "test"
        self.patch_paths = []
        self.patch_categories = []
        self.patch_names = []
        self.image_names = []
        self.image_loader = image_loader
        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # self.wide_label = np.load(f'/data_NUS_WIDE/NUS_{self.stage}_labels.npy')
        # if self.stage == "train":
        #     with open('/data_NUS_WIDE/TR', 'r') as f:
        #         image_paths = f.read().split('\n')[:-1]
        # elif self.stage == "test":
        #     with open('/data_NUS_WIDE/TE', 'r') as f:
        #         image_paths = f.read().split('\n')[:-1]

        # for i, image in enumerate(image_paths):
        #     cur_label = self.wide_label[i]
        #     dino_image_path = os.path.join(patch_root, image[1:])
        #     for patch_name in os.listdir(dino_image_path):
        #         if not patch_name.endswith('.jpg'):
        #             continue
        #         rank, logit = patch_name.split('_')[:2]
        #         if float(logit) > 0.15 or int(rank) == 1:
        #             self.patch_paths.append(os.path.join(dino_image_path, patch_name))
        #             self.patch_categories.append(cur_label)
        #             self.patch_names.append(patch_name)
        #             self.image_names.append(dino_image_path)

        i = 0
        labels = os.listdir(patch_root)
        for label in labels:
            file_root = os.path.join(patch_root, label)
            files = os.listdir(file_root)
            for file in files:
                cur_label = label
                i += 1
                patch_path = os.path.join(file_root, file)
                # ipdb.set_trace()
                for patch_name in os.listdir(patch_path):
                    if patch_name == file or not patch_name.endswith('.jpg'):
                        continue
                    rank, logit = patch_name.split('_')[:2]
                    rank = int(rank)
                    try:
                        logit = float(logit)
                    except ValueError:
                        ipdb.set_trace()
                    if logit > 0 or rank == 1:
                        self.patch_paths.append(os.path.join(patch_path, patch_name))
                        # self.patch_categories.append(label)
                        self.patch_categories.append(cur_label)
                        self.patch_names.append(patch_name)
                        self.image_names.append(f'{label}/{file}')
        print(f'length of patches: {len(self.patch_paths)}')

    def __getitem__(self, index):
        # ipdb.set_trace()
        patch_path = self.patch_paths[index]
        label = self.patch_categories[index]
        label = torch.tensor([int(label)])
        patch_name = self.patch_names[index]
        image_name = self.image_names[index]
        r = self.image_loader(patch_path)
        r = self.transform(r)
        return {
            'patch': r,
            'label': label,
            'patch_name': patch_name,
            'image_name': image_name,
        }

    def __len__(self):
        return len(self.patch_paths)


if __name__ == '__main__':
    import os
    os.sys.path.append('../')
    from models.bert import BertModel, BertTokenizer
    from opts.patch_opt import args
    args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # load data
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    # ipdb.set_trace()
    train_loader = build_dataset('train', tokenizer, args)

    start = time.time()
    for batch in train_loader:
        print(f"{time.time() - start}")
        time.sleep(4)  # other program
        start = time.time()
