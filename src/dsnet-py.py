import math
import numpy as np
import torch
from torch import nn
from pathlib import Path
from typing import List, Tuple

from anchor_based.dsnet import DSNet
from anchor_free.dsnet_af import DSNetAF
from helpers import data_helper, vsumm_helper, bbox_helper
from modules.model_zoo import get_model

class AnchorHelper:
    def get_anchors(seq_len: int, scales: List[int]):
        """Generate all multi-scale anchors for a sequence in center-width format.

        :param seq_len: Sequence length.
        :param scales: List of bounding box widths.
        :return: All anchors in center-width format.
        """
        anchors = np.zeros((seq_len, len(scales), 2), dtype=np.int32)
        for pos in range(seq_len):
            for scale_idx, scale in enumerate(scales):
                anchors[pos][scale_idx] = [pos, scale]
        return anchors

    def offset2bbox(offsets: np.ndarray, anchors: np.ndarray):
        """Convert predicted offsets to CW bounding boxes.

        :param offsets: Predicted offsets.
        :param anchors: Sequence anchors.
        :return: Predicted bounding boxes.
        """
        offsets = offsets.reshape(-1, 2)
        anchors = anchors.reshape(-1, 2)

        offset_center, offset_width = offsets[:, 0], offsets[:, 1]
        anchor_center, anchor_width = anchors[:, 0], anchors[:, 1]

        # Tc = Oc * Aw + Ac
        bbox_center = offset_center * anchor_width + anchor_center
        # Tw = exp(Ow) * Aw
        bbox_width = np.exp(offset_width) * anchor_width

        bbox = np.vstack((bbox_center, bbox_width)).T
        return bbox


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.sqrt_d_k = math.sqrt(d_k)

    def forward(self, Q, K, V):
        attn = torch.bmm(Q, K.transpose(2, 1))
        attn = attn / self.sqrt_d_k

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        y = torch.bmm(attn, V)

        return y, attn

class AttentionExtractor(nn.Module):
    def __init__(self, num_head=8, num_feature=1024):
        super().__init__()
        self.num_head = num_head

        self.Q = nn.Linear(num_feature, num_feature, bias=False)
        self.K = nn.Linear(num_feature, num_feature, bias=False)
        self.V = nn.Linear(num_feature, num_feature, bias=False)

        self.d_k = num_feature // num_head
        self.attention = ScaledDotProductAttention(self.d_k)

        self.fc = nn.Sequential(
            nn.Linear(num_feature, num_feature, bias=False),
            nn.Dropout(0.5)
        )

    def forward(self, *x):
        x = x[0]
        _, seq_len, num_feature = x.shape  # [1, seq_len, 1024]
        K = self.K(x)  # [1, seq_len, 1024]
        Q = self.Q(x)  # [1, seq_len, 1024]
        V = self.V(x)  # [1, seq_len, 1024]

        K = K.view(1, seq_len, self.num_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(self.num_head, seq_len, self.d_k)
        Q = Q.view(1, seq_len, self.num_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(self.num_head, seq_len, self.d_k)
        V = V.view(1, seq_len, self.num_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(self.num_head, seq_len, self.d_k)

        y, attn = self.attention(Q, K, V)  # [num_head, seq_len, d_k]
        y = y.view(1, self.num_head, seq_len, self.d_k).permute(0, 2, 1, 3).contiguous().view(1, seq_len, num_feature)

        y = self.fc(y)

        return y

class DSNet(nn.Module):
    def __init__(self, base_model, num_feature, num_hidden, anchor_scales, num_head):
        super().__init__()
        self.anchor_scales = anchor_scales
        self.num_scales = len(anchor_scales)
        self.base_model = AttentionExtractor(num_head, num_feature)

        self.roi_poolings = [nn.AvgPool1d(scale, stride=1, padding=scale // 2) for scale in anchor_scales]

        self.layer_norm = nn.LayerNorm(num_feature)
        self.fc1 = nn.Sequential(
            nn.Linear(num_feature, num_hidden),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden)
        )
        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)

    def forward(self, x):
        _, seq_len, _ = x.shape
        out = self.base_model(x)
        out = out + x
        out = self.layer_norm(out)

        out = out.transpose(2, 1)
        pool_results = [roi_pooling(out) for roi_pooling in self.roi_poolings]
        out = torch.cat(pool_results, dim=0).permute(2, 0, 1)[:-1]

        out = self.fc1(out)

        pred_cls = self.fc_cls(out).sigmoid().view(seq_len, self.num_scales)
        pred_loc = self.fc_loc(out).view(seq_len, self.num_scales, 2)

        return pred_cls, pred_loc

    def predict(self, seq):
        seq_len = seq.shape[1]
        pred_cls, pred_loc = self(seq)

        pred_cls = pred_cls.cpu().numpy().reshape(-1)
        pred_loc = pred_loc.cpu().numpy().reshape((-1, 2))

        anchors = AnchorHelper.get_anchors(seq_len, self.anchor_scales)
        anchors = anchors.reshape((-1, 2))

        pred_bboxes = AnchorHelper.offset2bbox(pred_loc, anchors)
        pred_bboxes = bbox_helper.cw2lr(pred_bboxes)

        return pred_cls, pred_bboxes

class Parameter:
    def __init__(self):
        
        self.model = 'anchor-based'
        self.device = "cuda"
        self.seed = 12345
        self.splits = ["../splits/tvsum.yml", "../splits/summe.yml"]
        self.max_epoch = 300
        self.model_dir = "../models/pretrain_ab_basic/"
        self.log_file = "log.txt"
        self.lr = 5e-5
        self.weight_decay = 1e-5
        self.lambda_reg = 1.0
        self.nms_thresh = 0.5

        self.ckpt_path = None
        self.sample_rate = 15
        self.source = None
        self.save_path = None

        self.base_model = 'attention'
        self.num_head = 8
        self.num_feature = 1024
        self.num_hidden = 128

        self.neg_sample_ratio = 2.0
        self.incomplete_sample_ratio = 1.0
        self.pos_iou_thresh = 0.6
        self.neg_iou_thresh = 0.0
        self.incomplete_iou_thresh = 0.3
        self.anchor_scales = [4,8,16,32]

        self.lambda_ctr = 1.0
        self.cls_loss = 'focal'
        self.reg_loss = 'soft-iou'

    def get_model(self):
        if self.model == 'anchor-based':
            return DSNet(self.base_model, self.num_feature, self.num_hidden, self.anchor_scales, self.num_head)
        elif self.model == 'anchor-free':
            return DSNetAF(self.base_model, self.num_feature, self.num_hidden, self.num_head)
        else:
            print(f'Invalid model type {self.model_type}')

def evaluate(model, val_loader, nms_thresh, device):
    model.eval()
    stats = data_helper.AverageMeter('fscore', 'diversity')

    with torch.no_grad():
        for test_key, seq, _, cps, n_frames, nfps, picks, user_summary in val_loader:
            seq_len = len(seq)
            seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)

            pred_cls, pred_bboxes = model.predict(seq_torch)

            pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)

            pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, nms_thresh)
            pred_summ = vsumm_helper.bbox2summary(
                seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks)

            eval_metric = 'avg' if 'tvsum' in test_key else 'max'
            fscore = vsumm_helper.get_summ_f1score(
                pred_summ, user_summary, eval_metric)

            pred_summ = vsumm_helper.downsample_summ(pred_summ)
            diversity = vsumm_helper.get_summ_diversity(pred_summ, seq)
            stats.update(fscore=fscore, diversity=diversity)

    return stats.fscore, stats.diversity


args = Parameter()
model = DSNet(args.base_model, args.num_feature, args.num_hidden, args.anchor_scales, args.num_head)
model = model.eval().to(args.device)
# model = model.train(False).to(args.device)

for split_path in args.splits:
    split_path = Path(split_path)
    splits = data_helper.load_yaml(split_path)

    stats = data_helper.AverageMeter('fscore', 'diversity')

    for split_idx, split in enumerate(splits):
        ckpt_path = data_helper.get_ckpt_path(args.model_dir, split_path, split_idx)
        state_dict = torch.load(str(ckpt_path),map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)

        val_set = data_helper.VideoDataset(split['test_keys'])
        val_loader = data_helper.DataLoader(val_set, shuffle=False)

        fscore, diversity = evaluate(model, val_loader, args.nms_thresh, args.device)
        stats.update(fscore=fscore, diversity=diversity)

        print(f'{split_path.stem} split {split_idx}: diversity: ' f'{diversity:.4f}, F-score: {fscore:.4f}')

    print(f'{split_path.stem}: diversity: {stats.diversity:.4f}, ' f'F-score: {stats.fscore:.4f}')