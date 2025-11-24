import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import ipdb

class PositionEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(PositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = x + self.pos_embedding
        if self.dropout:
            out = self.dropout(out)
        return out


class MlpBlock(nn.Module):
    """ Transformer Feed-Forward Block """

    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):

        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.fc2(out)
        out = self.dropout2(out)
        return out


class LinearGeneral(nn.Module):
    def __init__(self, in_dim=(768,), feat_dim=(12, 64)):
        super(LinearGeneral, self).__init__()

        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        # ipdb.set_trace()
        a = torch.tensordot(x, self.weight, dims=dims) + self.bias
        return a


class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim ** 0.5
        #         ipdb.set_trace()
        self.query = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.key = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.value = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.out = LinearGeneral((self.heads, self.head_dim), (in_dim,))

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x, attn_mask=None):
        #         ipdb.set_trace()
        b, n, _ = x.shape

        q = self.query(x, dims=([2], [0]))  # [2,197,12,64]
        k = self.key(x, dims=([2], [0]))  # [2,197,12,64]
        v = self.value(x, dims=([2], [0]))  # [2,197,12,64]

        q = q.permute(0, 2, 1, 3)  # [2,12,197,64]
        k = k.permute(0, 2, 1, 3)  # [2,12,197,64]
        v = v.permute(0, 2, 1, 3)  # [2,12,197,64]

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [2,12,197,197]
        
        if attn_mask is not None:
            
            attn_weights = attn_weights + attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        out = torch.matmul(attn_weights, v)  # [2,12,197,64]
        out = out.permute(0, 2, 1, 3)  # [2,197,12,64]

        # ipdb.set_trace()
        out = self.out(out, dims=([2, 3], [0, 1]))  # [2,197,768]

        return out, attn_weights


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = SelfAttention(in_dim, heads=num_heads, dropout_rate=attn_dropout_rate)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        # ipdb.set_trace()
        residual = x
        out = self.norm1(x)
        out, attn_scores = self.attn(out, attn_mask=attn_mask)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out

        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out, attn_scores


class Encoder(nn.Module):
    def __init__(self, num_patches, emb_dim, mlp_dim, num_layers=12, num_heads=12, dropout_rate=0.1,
                 attn_dropout_rate=0.0):
        super(Encoder, self).__init__()

        # positional embedding
        self.pos_embedding = PositionEmbs(num_patches, emb_dim, dropout_rate)
        # encoder blocks
        in_dim = emb_dim
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBlock(in_dim, mlp_dim, num_heads, dropout_rate, attn_dropout_rate)
            self.encoder_layers.append(layer)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        out = self.pos_embedding(x)
        # ipdb.set_trace()
        for layer in self.encoder_layers:
            out, attn_scores = layer(out, attn_mask=attn_mask)
        out = self.norm(out)
        return out, attn_scores


class VisionTransformer(nn.Module):
    """ Vision Transformer """

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
                 use_classifier=False,
                 out_dim=None,
                 ):
        super(VisionTransformer, self).__init__()
        self.args = args
        h, w = image_size
        # embedding layer
        fh, fw = patch_size
        gh, gw = h // fh, w // fw
        num_patches = gh * gw
        self.num_patches = num_patches
        self.embedding = nn.Conv2d(3, emb_dim, kernel_size=(fh, fw), stride=(fh, fw))
        # cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.emb_dim = emb_dim

        # transformer
        self.transformer = Encoder(
            num_patches=num_patches,  # 16 * 16 = 196
            emb_dim=emb_dim,          # 768
            mlp_dim=mlp_dim,          # 3072
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate)

        self.get_updateModel(patch_size)
        if use_classifier:
            self.classifier = nn.Linear(emb_dim, args.num_class)
        else:
            self.classifier = None
        if out_dim:
            self.dim_trans = nn.Linear(emb_dim, out_dim, bias=False)
        else:
            self.dim_trans = None

    def forward(self, batch, attn_mask=None, training=True, return_logits=True):
        # ipdb.set_trace()
        if isinstance(batch, dict):
            x = batch['image']
        else:
            x = batch
        emb = self.embedding(x)  # (n, c, gh, gw)[2,768,14,14]
        emb = emb.permute(0, 2, 3, 1)  # (n, gh, hw, c)[2,14,14,768]
        b, h, w, c = emb.shape
        emb = emb.reshape(b, h * w, c)  # [2,196,768]

        # prepend class token
        # ipdb.set_trace()
        cls_token = self.cls_token.repeat(b, 1, 1)  # [2,1,768]
        emb = torch.cat([cls_token, emb], dim=1)  # [2,197,768]

        # transformer
        feat, attn_scores = self.transformer(emb, attn_mask)  # feat.shape = [2, 197, 768]   attn_scores: (ns, n_heads, 197, 197)
        all_patch_tokens = feat[:, 1:]
        # select_patch_tokens, selected_indices = self.top_k_selection(all_patch_tokens, attn_scores)

        cls_token = feat[:, 0]  # use cls token to classify
        
        if self.dim_trans:
            return self.dim_trans(cls_token)
        
        if not self.classifier:
            return cls_token
        
        logits = self.classifier(cls_token)
        
        if training and isinstance(batch, dict):
            loss = nn.CrossEntropyLoss()(logits, batch['label'].view(-1))
        else:
            loss = 0.0

        return cls_token, {"ce loss": loss}

    def top_k_selection(self, all_patch_embeddings, attention_map):
        # caution! patch 196 attn 197

        attention_map = attention_map.mean(axis=1)

        attention_cls_part = attention_map[:, 0, 1:]
        attention_cls_part = attention_cls_part.squeeze()

        sorted, indices = torch.sort(attention_cls_part, descending=True)
        Rv = self.args.Rv
        select_token = round(Rv * self.num_patches)
        selected_indices = indices[:, 0:select_token]
        selected_patch_embedding = []
        for i in range(selected_indices.size(0)):  # bs
            all_patch_embeddings_i = all_patch_embeddings[i, :, :].squeeze()
            top_k_embedding = torch.index_select(all_patch_embeddings_i, 0, selected_indices[i])
            top_k_embedding = top_k_embedding.unsqueeze(0)
            selected_patch_embedding.append(top_k_embedding)
        selected_patch_embedding = torch.cat(selected_patch_embedding, 0)

        return selected_patch_embedding, selected_indices

    def get_updateModel(self, patch_size):
        # ipdb.set_trace()
        if patch_size == (16, 16):
            path = r'/mnt/cross_modal/models/pretrained/vit.pth'
            pretrained_dict = torch.load(path, map_location='cpu')['state_dict']
            # print('load model: vit16')
        elif patch_size == (32, 32):
            path = r'./models/pretrained/imagenet21k+imagenet2012_ViT-B_32.pth'
            pretrained_dict = torch.load(path, map_location='cpu')['state_dict']
            print('load model: vit32')

        model_dict = self.state_dict()
        shared_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(shared_dict)
#         ipdb.set_trace()
#         p_emb = model_dict['transformer.pos_embedding.pos_embedding']
#         p_emb = torch.cat((p_emb[:, :1], p_emb[:, 1:].repeat(1, 5, 1)), dim=1)
#         model_dict['transformer.pos_embedding.pos_embedding'] = p_emb

        # print("ckpt key lens{}".format(len(shared_dict.keys())))
        self.load_state_dict(model_dict, strict=False)
        return self


if __name__ == '__main__':
    from mgcc_opt import args

    model = VisionTransformer(args, num_classes=172)
    print(model)
    state_dict = model.state_dict()

    for key, value in state_dict.items():
        print("{}: {}".format(key, value.shape))
    x = torch.randn((2, 3, 224, 224))
    out = model(x)

    print('out:', out.shape)
