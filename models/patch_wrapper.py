import torch
from mpmath import backlunds
from torch import nn
from models.vit_backbone import VisionTransformer, EncoderBlock
from models.bert import BertModel
import ipdb
from torch.nn.utils.rnn import pad_sequence
from models.utils.modules4transformer import EncoderLayer, LayerNorm
import torch.nn.functional as F
import math
import numpy as np


class GateModule(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=768, output_dim=1):
        super(GateModule, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        weights = self.mlp(x)  # bs, num_, 1
        weights = torch.sigmoid(weights.squeeze()).unsqueeze(-1)
        x = x * weights
        return x.mean(dim=1)


class PatchWrapper(VisionTransformer):
    def __init__(self,
                 args,
                 image_size=(224, 224),
                 patch_size=(16, 16),
                 emb_dim=768,
                 mlp_dim=3072,
                 num_heads=12,
                 num_layers=12,
                 num_classes=81,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1,
                 vit_classifier=False,
                 out_dim=None,
                 use_extra_image=False,
                 front_door=False,
                 intervention_classifier=True,
                 return_logit=False,):
        super().__init__(args,
                         image_size=image_size,
                         patch_size=patch_size,
                         emb_dim=emb_dim,
                         mlp_dim=mlp_dim,
                         num_heads=num_heads,
                         num_layers=num_layers,
                         attn_dropout_rate=attn_dropout_rate,
                         dropout_rate=dropout_rate,
                         use_classifier=vit_classifier,
                         out_dim=out_dim,
                         )
        self.return_logit = return_logit
        self.use_extra_image = use_extra_image
        if intervention_classifier:
            self.wrapper_classifier = Interventional_Classifier(args.num_class)
        else:
            self.wrapper_classifier = nn.Linear(768, args.num_class)

        self.attn_fusion = nn.MultiheadAttention(embed_dim=768, num_heads=12, dropout=0.1, batch_first=True)
        self.front_door = front_door
        if args.dataset == 'wide':
            prototype_path = 'features/feats/wide_patch_prototype_thresh85.pt'
        elif args.dataset == 'vireo':
            prototype_path = 'features/prototype_tensor.pt'
        if front_door:
            self.clusters = nn.Parameter(
                torch.load(prototype_path, map_location=f'cuda:{args.local_rank}'),
                requires_grad=True)
            self.clusters_attn = nn.MultiheadAttention(embed_dim=768, num_heads=12, dropout=0.1, batch_first=True)
        # self.clusters_attn = GateModule()

    #         self.fusion_layer = EncoderBlock(in_dim=768, mlp_dim=3072, num_heads=12)
    #         # load initial weights
    #         self.fusion_layer.load_state_dict(self.transformer.encoder_layers[-1].state_dict())

    def forward(self, batch, return_logits=True, training=True):
        # ipdb.set_trace()
        if self.use_extra_image:
            patches = batch['extra_patches']
            patch_num = batch['extra_patch_num']
            patch_mask = batch['extra_patch_mask']
        else:
            patches = batch['patches']
            patch_num = batch['patch_num']
            patch_mask = batch['patch_mask']
        bs, patch_len, c, h, w = patches.shape
        patches = [patches[i, :patch_num[i]] for i in range(bs)]
        patches = torch.cat(patches, dim=0)
        patch_feat = super().forward(patches)[0]
        patch_feat = torch.split(patch_feat, patch_num.tolist())
        zeros = torch.zeros((bs, patch_len, 768)).to(patches.device)

        feat = pad_sequence(patch_feat, batch_first=True)  # (bs, max_len, 768)
        max_len = feat.size(1)
        # fixed padding len
        zeros[:, :max_len, :] = feat
        feat = zeros
        mask = patch_mask.bool()

        # extended_mask = -10000.0 * mask.unsqueeze(1).unsqueeze(2)

        # use self_attention to fuse feature
        attn_feat, attn_score = self.attn_fusion(feat, feat, feat, key_padding_mask=mask)  # (bs, batch_max_len, 768)
        # use transformer block to fuse feature
        # feat = self.fusion_layer(feat, attn_mask=extended_mask)[0]   # (bs, batch_max_len, 768)

        attn_feat = [attn_feat[i, :patch_num[i]] for i in range(bs)]
        # simple average
        attn_feat = [f.mean(0) for f in attn_feat]
        attn_feat = torch.stack(attn_feat)

        if not self.front_door:
            if self.return_logit:
                return self.wrapper_classifier(attn_feat)
            if return_logits:
                return self.wrapper_classifier(attn_feat), {}
            else:
                return self.wrapper_classifier(attn_feat), attn_feat
        # use lgcam frontdoor
        cluster_feat = self.clusters.float().unsqueeze(0).repeat(bs, 1, 1)
        z = self.clusters_attn(attn_feat.unsqueeze(1), cluster_feat, cluster_feat)[0].squeeze()
        # z = self.clusters_attn(cluster_feat)
        attn_feat = attn_feat + z
        logits = self.wrapper_classifier(attn_feat)

        if self.return_logit:
            return logits

        if return_logits:
            return logits, {}
        else:
            return logits, attn_feat


class PreprocessedPatchWrapper(nn.Module):
    def __init__(self, args, intervention_classifier=True):
        super().__init__()

        if intervention_classifier:
            self.wrapper_classifier = Interventional_Classifier(args.num_class)
        else:
            self.wrapper_classifier = nn.Linear(768, args.num_class)

        self.attn_fusion = nn.MultiheadAttention(embed_dim=768, num_heads=12, dropout=0.1, batch_first=True)


    def forward(self, batch, return_logits=True, training=True):
        # ipdb.set_trace()
        feat = batch['patch_feats']
        patch_num = batch['patch_num']
        mask = batch['patch_mask'].bool()
        bs = feat.size(0)

        # use self_attention to fuse feature
        attn_feat, attn_score = self.attn_fusion(feat, feat, feat, key_padding_mask=mask)  # (bs, batch_max_len, 768)
        # use transformer block to fuse feature
        # feat = self.fusion_layer(feat, attn_mask=extended_mask)[0]   # (bs, batch_max_len, 768)

        attn_feat = [attn_feat[i, :patch_num[i]] for i in range(bs)]
        # simple average
        attn_feat = [f.mean(0) for f in attn_feat]
        attn_feat = torch.stack(attn_feat)

        if return_logits:
            return self.wrapper_classifier(attn_feat), {}
        else:
            return self.wrapper_classifier(attn_feat), attn_feat


def uniform_loss(logits, num_class):
    batch_size = logits.size(0)
    y_uniform = torch.full((batch_size, num_class), 1.0 / num_class).to(logits.device)
    logits = F.softmax(logits, dim=-1)
    kl_div = F.kl_div(logits.log(), y_uniform, reduction='batchmean')
    return kl_div


def contrastive_loss(features, f_labels, prototypes, t=0.5):
    prototypes = prototypes.to(features)
    a_norm = features / features.norm(dim=1)[:, None]
    b_norm = prototypes / prototypes.norm(dim=1)[:, None]
    sim_matrix = torch.exp(torch.mm(a_norm, b_norm.transpose(0, 1)) / t)
    f_labels = f_labels.squeeze()
    b_norm_selected = b_norm[f_labels]
    if b_norm_selected.dim() == 1:
        b_norm_selected = b_norm_selected.unsqueeze(0)
    pos_sim = torch.exp(torch.diag(torch.mm(a_norm, b_norm_selected.transpose(0, 1))) / t)
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss

def calculate_prob_loss(logits, target):
    pred_probs = torch.softmax(logits, dim=-1)
    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    loss = criterion(torch.log(pred_probs), target)
    return loss


class ConfounderWrapper(VisionTransformer):
    def __init__(self,
                 args,
                 image_size=(224, 224),
                 patch_size=(16, 16),
                 emb_dim=768,
                 mlp_dim=3072,
                 num_heads=12,
                 num_layers=12,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1,
                 vit_classifier=False,
                 out_dim=None, ):
        super().__init__(args,
                         image_size=image_size,
                         patch_size=patch_size,
                         emb_dim=emb_dim,
                         mlp_dim=mlp_dim,
                         num_heads=num_heads,
                         num_layers=num_layers,
                         attn_dropout_rate=attn_dropout_rate,
                         dropout_rate=dropout_rate,
                         use_classifier=vit_classifier,
                         out_dim=out_dim)

        # self.wrapper_classifier = Interventional_Classifier(args.num_class)

        # self.context_clf = Interventional_Classifier(args.num_class)
        self.logit_clf = Interventional_Classifier(args.num_class, feat_dim=768)
        # self.frontdoor_clf = Interventional_Classifier(args.num_class)
        self.attn_fusion = nn.MultiheadAttention(embed_dim=768, num_heads=12, dropout=0.1, batch_first=True)
        self.rare_fusion = nn.MultiheadAttention(embed_dim=768, num_heads=12, dropout=0.1, batch_first=True)

        # self.clusters = nn.Parameter(
        #     torch.load('features/prototype_tensor.pt', map_location=f'cuda:{args.local_rank}'),
        #     requires_grad=True)
        # self.clusters_attn = nn.MultiheadAttention(embed_dim=768, num_heads=12, dropout=0.1, batch_first=True)

        # self.confounder_filter = nn.MultiheadAttention(embed_dim=768, num_heads=12, dropout=0.1, batch_first=True)
        # self.confounder_attn = nn.MultiheadAttention(embed_dim=768, num_heads=12, dropout=0.1, batch_first=True)
        # self.confounder_filter = nn.Linear(10, 1)
        # self.selector = nn.Linear(768, 768)
        # self.confounder = nn.Parameter(torch.load('dict/confounder_dict.pt', map_location=f'cuda:{args.local_rank}'), requires_grad=True)
        # self.confounder = nn.Parameter(torch.load('features/confounder_tensor.pt', map_location=f'cuda:{args.local_rank}'), requires_grad=True)

    def forward(self, batch, return_logits=True, training=True):
        patches, patch_num, patch_mask, rare_mask = batch['patches'], batch['patch_num'], batch['patch_mask'], batch[
            'rare_mask']
        # rare_num = rare_mask.size(1) - rare_mask.sum(1)
        # confounder, confounder_mask, confounder_num = batch['confounder'], batch['confounder_mask'], batch['confounder_num']
        # confounder = confounder.float()   # (bs, 10, 768)

        bs, patch_len, c, h, w = patches.shape
        patches = [patches[i, :patch_num[i]] for i in range(bs)]
        patches = torch.cat(patches, dim=0)
        patch_feat = super().forward(patches)[0]
        patch_feat = torch.split(patch_feat, patch_num.tolist())

        patch_feat = pad_sequence(patch_feat, batch_first=True)  # (bs, max_len, 768)
        max_len = patch_feat.size(1)
        patch_feat = F.pad(patch_feat, (0, 0, 0, patch_len - max_len))
        patch_mask = patch_mask.bool()

        # intervention
        # confounder_mask = confounder_mask.bool()
        # mask = patch_mask.unsqueeze(-1) | confounder_mask.unsqueeze(1)
        # mask = mask.unsqueeze(1).repeat(1, 12, 1, 1).view(bs * 12, mask.shape[1], mask.shape[2])
        # # caution! avoid setting whole line of attn_mask True, causing NAN
        # float_mask = torch.ones_like(mask, device=mask.device) * 1e-9 * mask

        # use attention to handle confounder
        # confounder, _ = self.confounder_filter(patch_feat, confounder, confounder, attn_mask=float_mask)
        # confounder_list = [confounder[i, :confounder_num[i]] for i in range(bs)]
        # # simple average
        # confounder = [f.mean(0) for f in confounder_list]
        # confounder = torch.stack(confounder)

        # use attention ??
        # confounder, _ = self.confounder_attn(confounder, confounder, confounder, key_padding_mask=confounder_mask)
        # use Linear to handle confounder
        # confounder = self.confounder_filter(confounder.transpose(1, 2).contiguous()).squeeze()

        # use self_attention to fuse feature
        n_patch_feat, attn_score = self.attn_fusion(patch_feat, patch_feat, patch_feat,
                                                    key_padding_mask=patch_mask)  # (bs, batch_max_len, 768)
        patch_feat_list = [n_patch_feat[i, :patch_num[i]] for i in range(bs)]
        # simple average
        avg_patch_feat = [f.mean(0) for f in patch_feat_list]
        avg_patch_feat = torch.stack(avg_patch_feat)

        # 1.feature subtract
        # ipdb.set_trace()
        # pre_logits = nn.Softmax(dim=1)(self.context_clf(avg_patch_feat))
        # confounder = torch.mm(pre_logits, self.confounder)
        # selector = self.selector(avg_patch_feat.clone())  # bs, 768
        # selector = selector.tanh()
        # backdoor_feat = avg_patch_feat - selector * self.confounder
        # backdoor_logits =  self.logit_clf(backdoor_feat)
        # return backdoor_logits

        # 2.logits subtract
        # ipdb.set_trace()
        # logits = self.logit_clf(patch_feat)
        # confounder = torch.mm(logits, self.confounder)
        # selector = self.selector(patch_feat.clone())
        # selector = selector.tanh()
        # context_logits = self.context_clf(selector * confounder)
        # return logits - context_logits

        # patch_derived confounder -> cross attention
        # confounder = self.confounder.float().unsqueeze(0).repeat(bs, 1, 1)
        # confounder = self.confounder_attn(avg_patch_feat.unsqueeze(1), confounder, confounder)[0].squeeze()
        # return self.logit_clf(avg_patch_feat - confounder)

        # front-door intervention
        # cluster_feat = self.clusters.float().unsqueeze(0).repeat(bs, 1, 1)
        # z = self.clusters_attn(avg_patch_feat.unsqueeze(1), cluster_feat, cluster_feat)[0].squeeze()
        # front_door_feature = avg_patch_feat + z
        # frontdoor_logits = self.frontdoor_clf(front_door_feature)
        # return backdoor_logits + frontdoor_logits

        # 3.confounder patches: adaptive confounder feat extractor
        # ipdb.set_trace()
        rare_feat, _ = self.rare_fusion(patch_feat, patch_feat, patch_feat, key_padding_mask=patch_mask)
        rare_num = patch_num
        rare_feat_list = [rare_feat[i, :rare_num[i]] for i in range(bs)]
        avg_rare_feat = [f.mean(0) for f in rare_feat_list]
        avg_rare_feat = torch.stack(avg_rare_feat)
        backdoor_feat = avg_patch_feat - avg_rare_feat
        ori_logits = self.logit_clf(avg_patch_feat)

        # 4.use loss to constrain
        if training:
            ori_loss = F.cross_entropy(ori_logits, batch['label'])
            # typical_contrast_loss = contrastive_loss(feat, batch['label'], batch['avg_category_typical_feat'][0], t=0.07)
            rare_logits = self.logit_clf(avg_rare_feat)
            r_loss = calculate_prob_loss(rare_logits, batch['confusion_prob'])
            # u_loss = uniform_loss(rare_logits, num_class=172)
        else:
            r_loss = ori_loss = 0
        # combined_feat = torch.cat((front_door_feature, backdoor_feat), dim=1)
        # ipdb.set_trace()
        # ipdb.set_trace()      typical_contrast_loss + u_loss
        logits = self.logit_clf(backdoor_feat)
        if return_logits:
            return logits, {'r_loss': r_loss}
            return logits, {'r_loss': r_loss, 'ori_loss': ori_loss}
        else:
            return backdoor_feat


class PatchLevelConfounderWrapper(VisionTransformer):
    def __init__(self,
                 args,
                 image_size=(224, 224),
                 patch_size=(16, 16),
                 emb_dim=768,
                 mlp_dim=3072,
                 num_heads=12,
                 num_layers=12,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1,
                 vit_classifier=False,
                 out_dim=None,
                 use_text=False):
        super().__init__(args,
                         image_size=image_size,
                         patch_size=patch_size,
                         emb_dim=emb_dim,
                         mlp_dim=mlp_dim,
                         num_heads=num_heads,
                         num_layers=num_layers,
                         attn_dropout_rate=attn_dropout_rate,
                         dropout_rate=dropout_rate,
                         use_classifier=vit_classifier,
                         out_dim=out_dim)

        self.logit_clf = Interventional_Classifier(args.num_class, feat_dim=768, num_head=4, tau=32.0)
        # self.logit_clf = Interventional_Classifier(args.num_class, feat_dim=768 * 2, num_head=4 * 2, tau=32.0 * 2)
        self.patch_relation_attn = nn.MultiheadAttention(embed_dim=768, num_heads=12, dropout=0.1, batch_first=True)
        self.confusing_attn = nn.MultiheadAttention(embed_dim=768, num_heads=12, dropout=0.1, batch_first=True)

        # with torch.no_grad():  # 在进行初始化时关闭梯度计算
        #     # 设置输入投影权重和偏置为0
        #     self.confusing_attn.in_proj_weight.fill_(0)
        #     self.confusing_attn.in_proj_bias.fill_(0)
        #
        #     # 设置输出投影权重和偏置为0
        #     self.confusing_attn.out_proj.weight.fill_(0)
        #     self.confusing_attn.out_proj.bias.fill_(0)
        self.use_text = args.use_text
        if self.use_text:
            self.txt_embed = BertModel.from_pretrained(args.bert_path).embeddings.word_embeddings
            # embed_weight = nn.Parameter(torch.load('wide_word_embedding.pt'), requires_grad=True)
            # self.txt_embed.weight = embed_weight
            # self.cross_attn = CrossLayer(embed_dim=768, num_heads=12, ff_dim=3072, dropout=0.1)
            # self.w_clf = Interventional_Classifier(args.num_class, feat_dim=768)
            # self.patch_attn_for_word = nn.MultiheadAttention(embed_dim=768, num_heads=12, dropout=0.1, batch_first=True)
            encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward=3072, dropout=0.1)
            # self.aggr = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=nn.LayerNorm(768))
            self.aggr = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=None)

    def z_score(self, x: torch.Tensor):
        if len(x.shape) == 3:
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True)
            return (x - mean) / (std + 1e-8)
        else:
            raise NotImplementedError

    def forward(self, batch, return_logits=True, training=True):
        patches, patch_num, patch_mask = batch['patches'], batch['patch_num'], batch['patch_mask']

        # extract patch features
        bs, patch_len, c, h, w = patches.shape
        patches = [patches[i, :patch_num[i]] for i in range(bs)]
        patches = torch.cat(patches, dim=0)
        patch_feat = super().forward(patches)
        patch_feat = torch.split(patch_feat, patch_num.tolist())
        patch_feat = pad_sequence(patch_feat, batch_first=True)  # (bs, max_len, 768)
        max_len = patch_feat.size(1)
        patch_feat = F.pad(patch_feat, (0, 0, 0, patch_len - max_len))
        # patch_mask = patch_mask.bool()

        # modeling patch relationship
        # ipdb.set_trace()
        n_patch_feat, attn_score = self.patch_relation_attn(patch_feat, patch_feat, patch_feat, key_padding_mask=patch_mask.bool())  # (bs, batch_max_len, 768)
        patch_feat_list = [n_patch_feat[i, :patch_num[i]] for i in range(bs)]
        # avg pooling
        avg_patch_feat = [f.mean(0) for f in patch_feat_list]
        avg_patch_feat = torch.stack(avg_patch_feat)
        # logits = self.logit_clf(avg_patch_feat)

        # ----------------------------------------------------
        # ipdb.set_trace()
        # prototype_feat, prototype_mask = batch['prototype_features'], batch['prototype_features_mask']
        confusing_feat, confusing_mask = batch['confusing_features'], batch['confusing_features_mask']
        confusing_label = batch['confusing_label']

        # avg_prototype_feat = prototype_feat.sum(1) / (1 - prototype_mask).sum(1).unsqueeze(1)
        # no_prototype = torch.all(avg_prototype_feat == 0, dim=1)
        # # if no prototype, no constraint
        # avg_prototype_feat[no_prototype] = avg_patch_feat[no_prototype]
        # prototype_loss = nn.MSELoss()(avg_prototype_feat, avg_patch_feat)
        # ipdb.set_trace()
        n_confusing_feat, _ = self.confusing_attn(confusing_feat, confusing_feat, confusing_feat, key_padding_mask=confusing_mask.bool())
        avg_confusing_feat = (n_confusing_feat * (1 - confusing_mask).unsqueeze(2)).sum(1) / (1 - confusing_mask).sum(1).unsqueeze(1)

        # n_confusing_feat, _ = self.confusing_attn(patch_feat, patch_feat, patch_feat, key_padding_mask=patch_mask.bool())
        # avg_confusing_feat = (n_confusing_feat * (1 - patch_mask).unsqueeze(2)).sum(1) / (1 - patch_mask).sum(1).unsqueeze(1)
        # ipdb.set_trace()
        if training:
            confusing_logits = self.logit_clf(avg_confusing_feat + 1e-9)
            # confusing_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(40)] * 81).cuda())(confusing_logits, confusing_label)
            confusing_loss = nn.BCEWithLogitsLoss()(confusing_logits, confusing_label)

            if confusing_loss < 0:
                ipdb.set_trace()
        else:
            confusing_loss = 0

        backdoor_feat = avg_patch_feat - avg_confusing_feat
        if not self.use_text:
            if return_logits:
                return self.logit_clf(backdoor_feat), {}
                return self.logit_clf(backdoor_feat), {'confusing_loss': 0.1 * confusing_loss}
            else:
                return self.logit_clf(backdoor_feat), backdoor_feat, avg_confusing_feat, avg_patch_feat

        # dual-direction cross-attention to use text-info
        # patch_feat_for_word, _ = self.patch_attn_for_word(patch_feat, patch_feat, patch_feat, key_padding_mask=patch_mask)
        # patch_feat_for_word_list = [patch_feat_for_word[i, :patch_num[i]] for i in range(bs)]
        # avg_patch_feat_for_word = [f.mean(0) for f in patch_feat_for_word_list]
        # avg_patch_feat_for_word = torch.stack(avg_patch_feat_for_word)

        # we must use text info !!!!
        # ipdb.set_trace()
        text_tokens = batch['pred_text_tokens']
        text_embeddings = self.txt_embed(text_tokens)
        text_mask = batch['pred_text_mask']  # 1 for mask

        text_embeddings = self.z_score(text_embeddings)
        n_patch_feat = self.z_score(n_patch_feat)

        fuse_feat = torch.cat((text_embeddings, n_patch_feat), dim=1)
        fuse_mask = torch.cat(((1 - text_mask).bool(), patch_mask.bool()), dim=1)
        fuse_feat = fuse_feat.transpose(0, 1)
        fuse_feat = self.aggr(fuse_feat, src_key_padding_mask=fuse_mask)
        fuse_feat = fuse_feat.transpose(0, 1)

        fuse_mask = (1 - fuse_mask.int()).unsqueeze(-1)
        fuse_feat = (fuse_feat * fuse_mask).sum(dim=1) / fuse_mask.sum(dim=1).float()

        if return_logits:
            return self.logit_clf(backdoor_feat + fuse_feat), {}
            # return self.logit_clf(torch.cat((backdoor_feat, fuse_feat), dim=1)), {}
        else:
            return self.logit_clf(backdoor_feat + fuse_feat), backdoor_feat + fuse_feat

        # vis_mask = 1 - batch['patch_mask'].int()  # 1 for mask
        # attn_mask = text_mask.unsqueeze(-1) & vis_mask.unsqueeze(1)
        # fuse_feat = self.cross_attn(text_embeddings, patch_feat_for_word, attn_mask)
        # fuse_feat = (fuse_feat * text_mask.unsqueeze(-1)).sum(dim=1) / text_mask.unsqueeze(-1).sum(dim=1).float()
        #
        # fuse_feat = avg_patch_feat_for_word + fuse_feat
        # w_logits = self.w_clf(fuse_feat)
        # return self.logit_clf(backdoor_feat) + w_logits, {}
        #
        # feat = torch.cat((backdoor_feat, fuse_feat), dim=1)

        return self.logit_clf(backdoor_feat), {'confusing_loss': 0.05 * confusing_loss,}
        return self.logit_clf(feat), {'confusing_loss': confusing_loss, 'prototype_loss': prototype_loss}



class DualPathFuseModule(VisionTransformer):
    def __init__(self,
                 args,
                 image_size=(224, 224),
                 patch_size=(16, 16),
                 emb_dim=768,
                 mlp_dim=3072,
                 num_heads=12,
                 num_layers=12,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1,
                 vit_classifier=False,
                 out_dim=None, ):
        super().__init__(args,
                         image_size=image_size,
                         patch_size=patch_size,
                         emb_dim=emb_dim,
                         mlp_dim=mlp_dim,
                         num_heads=num_heads,
                         num_layers=num_layers,
                         attn_dropout_rate=attn_dropout_rate,
                         dropout_rate=dropout_rate,
                         use_classifier=vit_classifier,
                         out_dim=out_dim)

        self.txt_embed = BertModel.from_pretrained(args.bert_path).embeddings.word_embeddings
        self.logit_clf = Interventional_Classifier(args.num_class)
        # word-match
        # self.cross_attn = CrossLayer(embed_dim=768, num_heads=12, ff_dim=3072, dropout=0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward=3072, dropout=0.1)
        # self.aggr = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=nn.LayerNorm(768))
        self.aggr = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=None)
        # patch-fuse
        self.attn_fusion = nn.MultiheadAttention(embed_dim=768, num_heads=12, dropout=0.1, batch_first=True)

    def z_score(self, x: torch.Tensor):
        if len(x.shape) == 3:
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True)
            return (x - mean) / (std + 1e-8)
        else:
            raise NotImplementedError


    def forward(self, batch, return_logits=True, training=True):
        # ipdb.set_trace()
        patches = batch['patches']
        bs, patch_len, c, h, w = patches.shape
        patches = [patches[i, :batch['patch_num'][i]] for i in range(bs)]
        patches = torch.cat(patches, dim=0)
        patch_feat = super().forward(patches)[0]
        patch_feat = torch.split(patch_feat, batch['patch_num'].tolist())
        zeros = torch.zeros((bs, patch_len, 768)).to(patches.device)
        # self-attention to fuse visual feature alone
        feat = pad_sequence(patch_feat, batch_first=True)  # (bs, max_len, 768)
        max_len = feat.size(1)
        # fixed padding len
        zeros[:, :max_len, :] = feat
        v_feat = zeros
        vis_mask = batch['patch_mask'].bool()
        seq_vis_feat, attn_score = self.attn_fusion(v_feat, v_feat, v_feat,
                                                    key_padding_mask=vis_mask)  # (bs, batch_max_len, 768)
        vis_feat = [seq_vis_feat[i, :batch['patch_num'][i]] for i in range(bs)]
        # simple average
        vis_feat = [f.mean(0) for f in vis_feat]
        vis_feat = torch.stack(vis_feat)

        #################################################
        # dual-direction cross-attention to use text-info
        # text_tokens = batch['pred_text_tokens']
        # text_embeddings = self.txt_embed(text_tokens)
        # # ipdb.set_trace()
        # vis_mask = 1 - batch['attn_mask'].int()  # 1 for mask
        # text_mask = batch['text_masks']  # 1 for mask
        #
        # attn_mask = text_mask.unsqueeze(-1) & vis_mask.unsqueeze(1)
        # fuse_feat = self.cross_attn(text_embeddings, seq_vis_feat, attn_mask)
        #
        # fuse_feat = (fuse_feat * text_mask.unsqueeze(-1)).sum(dim=1) / text_mask.unsqueeze(-1).sum(dim=1).float()
        #
        # logits = self.w_classifier(vis_feat + fuse_feat)
        # # logits = self.w_classifier(fuse_feat)
        # return logits

        text_tokens = batch['pred_text_tokens']
        text_embeddings = self.txt_embed(text_tokens)
        text_mask = batch['pred_text_mask']  # 1 for mask

        text_embeddings = self.z_score(text_embeddings)
        n_patch_feat = self.z_score(seq_vis_feat)

        fuse_feat = torch.cat((text_embeddings, n_patch_feat), dim=1)
        fuse_mask = torch.cat(((1 - text_mask).bool(), vis_mask.bool()), dim=1)
        fuse_feat = fuse_feat.transpose(0, 1)
        fuse_feat = self.aggr(fuse_feat, src_key_padding_mask=fuse_mask)
        fuse_feat = fuse_feat.transpose(0, 1)

        fuse_mask = (1 - fuse_mask.int()).unsqueeze(-1)
        fuse_feat = (fuse_feat * fuse_mask).sum(dim=1) / fuse_mask.sum(dim=1).float()

        if return_logits:
            return self.logit_clf(vis_feat + fuse_feat), {}
            return self.logit_clf(torch.cat((backdoor_feat, fuse_feat), dim=1)), {}
        else:
            return backdoor_feat + fuse_feat

        #################################################
        # concat visual and textual feature, then fusing

        # vis_mask = batch['attn_mask'].bool()   # True for mask
        # text_mask = (1 - batch['text_masks']).bool()
        #
        # fuse_feat = torch.cat([seq_vis_feat, text_embeddings], dim=1)
        # fuse_mask = torch.cat([vis_mask, text_mask], dim=1)
        #
        # attn_feat, _ = self.attn_fusion(fuse_feat, fuse_feat, fuse_feat, key_padding_mask=fuse_mask)
        # attn_feat = (attn_feat * fuse_mask.unsqueeze(-1)).sum(dim=1) / fuse_mask.unsqueeze(-1).sum(dim=1).float()
        #
        # logits = self.w_classifier(attn_feat)
        # return logits

class CausalModule(nn.Module):
    def __init__(self, args):
        super(CausalModule, self).__init__()
        self.front_door_module = PatchWrapper(args, vit_classifier=False, front_door=True)
        # self.patch_relation_modeling = PatchWrapper(args, vit_classifier=False, intervention_classifier=False)
        # self.tde_simulator = ConfounderWrapper(args, vit_classifier=False)
        self.patch_level_confounder = PatchLevelConfounderWrapper(args, vit_classifier=False)
        self.base = VisionTransformer(args, use_classifier=False)
        self.classifier = Interventional_Classifier(args.num_class, feat_dim=768 * 3, num_head=4 * 3, tau=32.0 * 3)

    def forward(self, x, training=True, return_logits=True):
        feat_front_door = self.front_door_module(x, return_logits=False)[1]
        feat_patch_level_confounder = self.patch_level_confounder(x, return_logits=False)[1]
        # feat_patch_relation = self.patch_relation_modeling(x, return_logits=False)
        feat_base = self.base(x)[0]
        # ipdb.set_trace()
        return self.classifier(torch.cat((feat_front_door, feat_patch_level_confounder, feat_base), dim=1)), {}

        # return logits1 + logits2
        base_logit = self.base(x)[0]
        patch_relation_logits = self.patch_relation_modeling(x)[0]
        # ipdb.set_trace()
        return base_logit + patch_relation_logits, 0
        return logits1 + logits2 + base_logit

        # feat1 = self.front_door_module(x, return_logits=False)
        # feat2 = self.tde_simulator(x, return_logits=False)
        # feat = torch.cat((feat1, feat2), dim=1)
        # return self.classifier(feat)

class Interventional_Classifier(nn.Module):
    def __init__(self, num_classes=172, feat_dim=768, num_head=4, tau=32.0, beta=0.03125):
        super(Interventional_Classifier, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim).cuda(), requires_grad=True)
        self.scale = tau / num_head  # 32.0 / num_head
        self.norm_scale = beta  # 1.0 / 32.0    beta is alpha in the paper
        self.num_head = num_head
        self.head_dim = feat_dim // num_head
        self.reset_parameters(self.weight)
        self.feat_dim = feat_dim

    @staticmethod
    def reset_parameters(weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x_list = torch.split(x, self.head_dim, dim=1)
        w_list = torch.split(self.weight, self.head_dim, dim=1)
        y_list = []
        for x_, w_ in zip(x_list, w_list):
            normed_x = x_ / torch.norm(x_, 2, 1, keepdim=True)
            normed_w = w_ / (torch.norm(w_, 2, 1, keepdim=True) + self.norm_scale)
            y_ = torch.mm(normed_x * self.scale, normed_w.t())
            y_list.append(y_)
        y = sum(y_list)
        return y


class CrossLayer(EncoderLayer):
    """
    composed by a cross-attention layer and feed-forward network
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super(CrossLayer, self).__init__(embed_dim, num_heads, ff_dim, dropout)

    def forward(self, x, y, mask=None):
        x = self.sublayer_attn(x, lambda x: self.attn(x, y, y, mask))  # multi-head attention
        x = self.sublayer_ff(x, self.feed_forward)
        return x


class FeatureFuse(nn.Module):
    def __init__(self, embed_dim):
        super(FeatureFuse, self).__init__()
        self.embed_dim = embed_dim
        self.fuse_v = CrossLayer(embed_dim, 8, 4 * embed_dim, .1)
        self.fuse_t = CrossLayer(embed_dim, 8, 4 * embed_dim, .1)
        self.norm = LayerNorm(embed_dim)

    def forward(self, text, vis):
        # input: query  key_value
        mediator = self.fuse_t(vis, text)  # visual-linguistic fuse module
        mediator = self.fuse_v(mediator, vis) + mediator
        out = self.norm(mediator)
        return out