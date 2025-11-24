# coding: UTF-8
import os.path
from datetime import datetime
import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer


class BertWrapper(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(768, args.num_class)

    def forward(self, x):
        ipdb.set_trace()
        context = x[0]  # 输入的句子 shape (64, 32)
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0] shape (64, 32)
        # Outputs: Tuple of (encoded_layers, pooled_output)
        _, word_ft = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # pooled shape (batch, 768)

        mask = mask.unsqueeze(-1)
        word_ft = word_ft * mask.sum(dim=1) / mask.sum(dim=1).float()

        out = self.fc(word_ft)
        return out

