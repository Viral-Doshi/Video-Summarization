import random
import numpy as np
import torch
from torch import nn
import yaml
import math
import h5py
from ortools.algorithms.pywrapknapsack_solver import KnapsackSolver

from pathlib import Path
from itertools import groupby
from operator import itemgetter
from time import gmtime, strftime


class DataHelper:
    def load_yaml(path):
        with open(path) as f:
            obj = yaml.safe_load(f)
        return obj

    def dump_yaml(obj, path):
        with open(path, 'w') as f:
            yaml.dump(obj, f)

    def get_ckpt_path(model_dir, split_path, split_index):
        split_path = Path(split_path)
        return Path(model_dir) / 'checkpoint' / f'{split_path.name}.{split_index}.pt'

    def get_ckpt_dir(model_dir):
        return Path(model_dir) / 'checkpoint'

class Parameter:
    def __init__(self):
        self.model = 'anchor-based'
        self.device = "cuda"
        self.splits = ["../splits/summe.yml"]
        self.model_dir = "../models/ab_basic"
        self.nms_thresh = 0.5
        self.ckpt_path = None
        self.base_model = ['attention','bilstm'][0]
        self.num_head = 8
        self.num_feature = 1024
        self.num_hidden = 128
        self.lr = 5e-5
        self.weight_decay = 1e-5
        self.lambda_reg = 1.0
        self.nms_thresh = 0.5
        self.max_epoch = 300
        self.pos_iou_thresh = 0.6
        self.neg_iou_thresh = 0.0
        self.incomplete_iou_thresh = 0.3
        self.lambda_ctr = 1.0
        self.neg_sample_ratio = 2.0
        self.incomplete_sample_ratio = 1.0
        self.cls_loss = 'focal'
        self.reg_loss = 'soft-iou'
        self.anchor_scales = [4,8,16,32]
        self.log_file = 'log.txt'
        self.seed = 12345

class AverageMeter:
    def __init__(self, *keys):
        self.totals = {key: 0.0 for key in keys}
        self.counts = {key: 0 for key in keys}

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.totals[key] += value
            self.counts[key] += 1

    def __getattr__(self, attr):
        total = self.totals[attr]
        count = self.counts[attr]
        return total / count if count else 0.0

class AnchorHelper:
    def get_anchors(seq_len, scales):
        anchors = np.zeros((seq_len, len(scales), 2), dtype=np.int32)
        for pos in range(seq_len):
            for scale_idx, scale in enumerate(scales):
                anchors[pos][scale_idx] = [pos, scale]
        return anchors

    def offset2bbox(offsets, anchors):
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

    def bbox2offset(bboxes, anchors):
        bbox_center, bbox_width = bboxes[:, 0], bboxes[:, 1]
        anchor_center, anchor_width = anchors[:, 0], anchors[:, 1]
        # Oc = (Tc - Ac) / Aw
        offset_center = (bbox_center - anchor_center) / anchor_width
        # Ow = ln(Tw / Aw)
        offset_width = np.log(bbox_width / anchor_width)

        offset = np.vstack((offset_center, offset_width)).T
        return offset

    def get_pos_label(anchors,targets,iou_thresh):
        seq_len, num_scales, _ = anchors.shape
        anchors = np.reshape(anchors, (seq_len * num_scales, 2))

        loc_label = np.zeros((seq_len * num_scales, 2))
        cls_label = np.zeros(seq_len * num_scales, dtype=np.int32)

        for target in targets:
            target = np.tile(target, (seq_len * num_scales, 1))
            iou = BboxHelper.iou_cw(anchors, target)
            pos_idx = np.where(iou > iou_thresh)
            cls_label[pos_idx] = 1
            loc_label[pos_idx] = AnchorHelper.bbox2offset(target[pos_idx], anchors[pos_idx])

        loc_label = loc_label.reshape((seq_len, num_scales, 2))
        cls_label = cls_label.reshape((seq_len, num_scales))

        return cls_label, loc_label

    def get_neg_label(cls_label, num_neg):
        seq_len, num_scales = cls_label.shape
        cls_label = cls_label.copy().reshape(-1)
        cls_label[cls_label < 0] = 0  # reset negative samples

        neg_idx, = np.where(cls_label == 0)
        np.random.shuffle(neg_idx)
        neg_idx = neg_idx[:num_neg]

        cls_label[neg_idx] = -1
        cls_label = np.reshape(cls_label, (seq_len, num_scales))
        return cls_label

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.sqrt_d_k = math.sqrt(d_k)

    def forward(self, Q, K, V):

        attn = torch.bmm(Q, K.transpose(2, 1))
        # print("attn: ",attn.shape)
        attn = attn / self.sqrt_d_k
        # print("scaled attn: ",attn.shape)

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

    def forward(self, x):
        _, seq_len, num_feature = x.shape  # [1, seq_len, 1024]
        K = self.K(x)  # [1, seq_len, 1024]
        Q = self.Q(x)  # [1, seq_len, 1024]
        V = self.V(x)  # [1, seq_len, 1024]

        K = K.view(1, seq_len, self.num_head, self.d_k).permute(2, 0, 1, 3).view(self.num_head, seq_len, self.d_k)
        Q = Q.view(1, seq_len, self.num_head, self.d_k).permute(2, 0, 1, 3).view(self.num_head, seq_len, self.d_k)
        V = V.view(1, seq_len, self.num_head, self.d_k).permute(2, 0, 1, 3).view(self.num_head, seq_len, self.d_k)

        y, attn = self.attention(Q, K, V)  # [num_head, seq_len, d_k]
        y = y.view(1, self.num_head, seq_len, self.d_k).permute(0, 2, 1, 3).contiguous().view(1, seq_len, num_feature)

        y = self.fc(y)

        return y

class LSTMExtractor(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *inputs):
        out, _ = super().forward(*inputs)
        return out

class BboxHelper:
    def cw2lr(bbox_cw):
        bbox_cw = np.asarray(bbox_cw, dtype=np.float32).reshape((-1, 2))
        left = bbox_cw[:, 0] - bbox_cw[:, 1] / 2
        right = bbox_cw[:, 0] + bbox_cw[:, 1] / 2
        bbox_lr = np.vstack((left, right)).T
        return bbox_lr

    def lr2cw(bbox_lr):
        bbox_lr = np.asarray(bbox_lr, dtype=np.float32).reshape((-1, 2))
        center = (bbox_lr[:, 0] + bbox_lr[:, 1]) / 2
        width = bbox_lr[:, 1] - bbox_lr[:, 0]
        bbox_cw = np.vstack((center, width)).T
        return bbox_cw

    def seq2bbox(sequence):
        sequence = np.asarray(sequence, dtype=np.bool)
        selected_indices, = np.where(sequence == 1)

        bboxes_lr = []
        for k, g in groupby(enumerate(selected_indices), lambda x: x[0] - x[1]):
            segment = list(map(itemgetter(1), g))
            start_frame, end_frame = segment[0], segment[-1] + 1
            bboxes_lr.append([start_frame, end_frame])

        bboxes_lr = np.asarray(bboxes_lr, dtype=np.int32)
        return bboxes_lr

    def iou_lr(anchor_bbox, target_bbox):
        anchor_left, anchor_right = anchor_bbox[:, 0], anchor_bbox[:, 1]
        target_left, target_right = target_bbox[:, 0], target_bbox[:, 1]

        inter_left = np.maximum(anchor_left, target_left)
        inter_right = np.minimum(anchor_right, target_right)
        union_left = np.minimum(anchor_left, target_left)
        union_right = np.maximum(anchor_right, target_right)

        intersect = inter_right - inter_left
        intersect[intersect < 0] = 0
        union = union_right - union_left
        union[union <= 0] = 1e-6

        iou = intersect / union
        return iou

    def iou_cw(anchor_bbox, target_bbox):
        anchor_bbox_lr = BboxHelper.cw2lr(anchor_bbox)
        target_bbox_lr = BboxHelper.cw2lr(target_bbox)
        return BboxHelper.iou_lr(anchor_bbox_lr, target_bbox_lr)

    def nms(scores, bboxes, thresh):

        valid_idx = bboxes[:, 0] < bboxes[:, 1]
        scores = scores[valid_idx]
        bboxes = bboxes[valid_idx]
        arg_desc = scores.argsort()[::-1]

        scores_remain = scores[arg_desc]
        bboxes_remain = bboxes[arg_desc]

        keep_bboxes = []
        keep_scores = []

        while bboxes_remain.size > 0:
            bbox = bboxes_remain[0]
            score = scores_remain[0]
            keep_bboxes.append(bbox)
            keep_scores.append(score)

            iou = BboxHelper.iou_lr(bboxes_remain, np.expand_dims(bbox, axis=0))

            keep_indices = (iou < thresh)
            bboxes_remain = bboxes_remain[keep_indices]
            scores_remain = scores_remain[keep_indices]

        keep_bboxes = np.asarray(keep_bboxes, dtype=bboxes.dtype)
        keep_scores = np.asarray(keep_scores, dtype=scores.dtype)

        return keep_scores, keep_bboxes

class VSummHelper:
    def f1_score(pred , test):
        assert pred.shape == test.shape
        pred = np.asarray(pred, dtype=np.bool)
        test = np.asarray(test, dtype=np.bool)
        overlap = (pred & test).sum()
        if overlap == 0:
            return 0.0
        precision = overlap / pred.sum()
        recall = overlap / test.sum()
        f1 = 2 * precision * recall / (precision + recall)
        return float(f1)

    def knapsack(values , weights, capacity):
        knapsack_solver = KnapsackSolver( KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER, 'test')

        values = list(values)
        weights = list(weights)
        capacity = int(capacity)

        knapsack_solver.Init(values, [weights], [capacity])
        knapsack_solver.Solve()
        packed_items = [x for x in range(0, len(weights)) if knapsack_solver.BestSolutionContains(x)]
        return packed_items

    def downsample_summ(summ):
        return summ[::15]

    def get_keyshot_summ(pred, cps, n_frames, nfps, picks, proportion = 0.15):
        assert pred.shape == picks.shape
        picks = np.asarray(picks, dtype=np.int32)

        frame_scores = np.zeros(n_frames, dtype=np.float32)
        for i in range(len(picks)):
            pos_lo = picks[i]
            pos_hi = picks[i + 1] if i + 1 < len(picks) else n_frames
            frame_scores[pos_lo:pos_hi] = pred[i]

        seg_scores = np.zeros(len(cps), dtype=np.int32)
        for seg_idx, (first, last) in enumerate(cps):
            scores = frame_scores[first:last + 1]
            seg_scores[seg_idx] = int(1000 * scores.mean())

        limits = int(n_frames * proportion)
        packed = VSummHelper.knapsack(seg_scores, nfps, limits)

        summary = np.zeros(n_frames, dtype=np.bool)
        for seg_idx in packed:
            first, last = cps[seg_idx]
            summary[first:last + 1] = True

        return summary

    def bbox2summary(seq_len, pred_cls, pred_bboxes, change_points, n_frames, nfps, picks):
        score = np.zeros(seq_len, dtype=np.float32)
        for bbox_idx in range(len(pred_bboxes)):
            lo, hi = pred_bboxes[bbox_idx, 0], pred_bboxes[bbox_idx, 1]
            score[lo:hi] = np.maximum(score[lo:hi], [pred_cls[bbox_idx]])

        pred_summ = VSummHelper.get_keyshot_summ(score, change_points, n_frames, nfps, picks)
        return pred_summ

    def get_summ_f1score(pred_summ, test_summ, eval_metric = 'avg'):

        pred_summ = np.asarray(pred_summ, dtype=np.bool)
        test_summ = np.asarray(test_summ, dtype=np.bool)
        _, n_frames = test_summ.shape
        if pred_summ.size > n_frames:
            pred_summ = pred_summ[:n_frames]
        elif pred_summ.size < n_frames:
            pred_summ = np.pad(pred_summ, (0, n_frames - pred_summ.size))
        f1s = [VSummHelper.f1_score(user_summ, pred_summ) for user_summ in test_summ]

        if eval_metric == 'avg':
            final_f1 = np.mean(f1s)
        elif eval_metric == 'max':
            final_f1 = np.max(f1s)
        else:
            raise ValueError(f'Invalid eval metric {eval_metric}')

        return float(final_f1)

class DSNet(nn.Module):
    def __init__(self, base_model, num_feature, num_hidden, anchor_scales, num_head):
        super().__init__()
        self.anchor_scales = anchor_scales
        self.num_scales = len(anchor_scales)
        if base_model == 'attention':
            self.base_model = AttentionExtractor(num_head, num_feature)
        else: #put bilstm
            self.base_model = LSTMExtractor(num_feature, num_feature // 2, bidirectional=True)

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
        # print("0 GoogleNet Features(v): ",x.shape)
        out = self.base_model(x)
        # print("1 Temporal Layer Featurs(w): ",out.shape)
        out = out + x
        # print("2 Final Features(x): ", out.shape)
        out = self.layer_norm(out)
        # print("3 After Layer Norm: ",out.shape)
        out = out.transpose(2, 1)
        # print("4 After Transpose: ",out.shape)
        pool_results = [roi_pooling(out) for roi_pooling in self.roi_poolings]
        # print("5 After AvgPooling: ",len(pool_results), len(pool_results[0]), len(pool_results[0][0]), len(pool_results[0][0][0]))
        out = torch.cat(pool_results, dim=0).permute(2, 0, 1)[:-1]
        # print("6 After ConCat: ",out.shape)

        out = self.fc1(out)

        pred_cls = self.fc_cls(out).sigmoid().view(seq_len, self.num_scales)
        pred_loc = self.fc_loc(out).view(seq_len, self.num_scales, 2)

        return pred_cls, pred_loc

    def predict(self, seq):
        seq_len = seq.shape[1]
        pred_cls, pred_loc = self(seq)
        # print("pred_loc", pred_loc)
        pred_cls = pred_cls.cpu().numpy().reshape(-1)
        pred_loc = pred_loc.cpu().numpy().reshape((-1, 2))

        anchors = AnchorHelper.get_anchors(seq_len, self.anchor_scales)
        anchors = anchors.reshape((-1, 2))

        pred_bboxes = AnchorHelper.offset2bbox(pred_loc, anchors)
        pred_bboxes = BboxHelper.cw2lr(pred_bboxes)

        return pred_cls, pred_bboxes

class VideoDataset:
    def __init__(self, keys):
        self.keys = keys
        self.datasets = VideoDataset.get_datasets(keys)

    def __getitem__(self, index):
        key = self.keys[index]
        video_path = Path(key)
        dataset_name = str(video_path.parent)
        video_name = video_path.name
        video_file = self.datasets[dataset_name][video_name]

        seq = video_file['features'][...].astype(np.float32)
        gtscore = video_file['gtscore'][...].astype(np.float32)
        cps = video_file['change_points'][...].astype(np.int32)
        n_frames = video_file['n_frames'][...].astype(np.int32)
        nfps = video_file['n_frame_per_seg'][...].astype(np.int32)
        picks = video_file['picks'][...].astype(np.int32)
        user_summary = None
        if 'user_summary' in video_file:
            user_summary = video_file['user_summary'][...].astype(np.float32)

        gtscore -= gtscore.min()
        gtscore /= gtscore.max()

        return key, seq, gtscore, cps, n_frames, nfps, picks, user_summary

    def __len__(self):
        return len(self.keys)

    def get_datasets(keys):
        dataset_paths = {str(Path(key).parent) for key in keys}
        datasets = {path: h5py.File(path, 'r') for path in dataset_paths}
        return datasets

class DataLoader:
    def __init__(self, dataset, shuffle):
        self.dataset = dataset
        self.shuffle = shuffle
        self.data_idx = list(range(len(self.dataset)))

    def __iter__(self):
        self.iter_idx = 0
        if self.shuffle:
            random.shuffle(self.data_idx)
        return self

    def __next__(self):
        if self.iter_idx == len(self.dataset):
            raise StopIteration
        curr_idx = self.data_idx[self.iter_idx]
        batch = self.dataset[curr_idx]
        self.iter_idx += 1
        return batch

def xavier_init(module):
    cls_name = module.__class__.__name__
    if 'Linear' in cls_name or 'Conv' in cls_name:
        nn.init.xavier_uniform_(module.weight, gain=np.sqrt(2.0))
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.1)

def evaluate(model, val_loader, nms_thresh, device):
    model.eval()
    stats = AverageMeter('fscore')

    with torch.no_grad():
        for test_key, seq, _, cps, n_frames, nfps, picks, user_summary in val_loader:
            seq_len = len(seq)
            seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)
            # print(model)
            pred_cls, pred_bboxes = model.predict(seq_torch)
            pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)
            pred_cls, pred_bboxes = BboxHelper.nms(pred_cls, pred_bboxes, nms_thresh)
            pred_summ = VSummHelper.bbox2summary(seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks)

            eval_metric = 'avg' if 'tvsum' in test_key else 'max'
            fscore = VSummHelper.get_summ_f1score(pred_summ, user_summary, eval_metric)
            pred_summ = VSummHelper.downsample_summ(pred_summ)
            stats.update(fscore=fscore)

    return stats.fscore

def calc_loc_loss(pred_loc,test_loc,cls_label,use_smooth = True):
    pos_idx = cls_label.eq(1).unsqueeze(-1).repeat((1, 1, 2))
    pred_loc = pred_loc[pos_idx]
    test_loc = test_loc[pos_idx]
    if use_smooth:
        loc_loss = nn.functional.smooth_l1_loss(pred_loc, test_loc)
    else:
        loc_loss = (pred_loc - test_loc).abs().mean()

    return loc_loss

def calc_cls_loss(pred, test):
    pred = pred.view(-1)
    test = test.view(-1)

    pos_idx = test.eq(1).nonzero().squeeze(-1)
    pred_pos = pred[pos_idx].unsqueeze(-1)
    pred_pos = torch.cat([1 - pred_pos, pred_pos], dim=-1)
    gt_pos = torch.ones(pred_pos.shape[0], dtype=torch.long, device=pred.device)
    loss_pos = nn.functional.nll_loss(pred_pos.log(), gt_pos)

    neg_idx = test.eq(-1).nonzero().squeeze(-1)
    pred_neg = pred[neg_idx].unsqueeze(-1)
    pred_neg = torch.cat([1 - pred_neg, pred_neg], dim=-1)
    gt_neg = torch.zeros(pred_neg.shape[0], dtype=torch.long, device=pred.device)
    loss_neg = nn.functional.nll_loss(pred_neg.log(), gt_neg)

    loss = (loss_pos + loss_neg) * 0.5
    return loss

def trainer(args, split, save_path):
    model = DSNet(base_model=args.base_model, num_feature=args.num_feature,
                  num_hidden=args.num_hidden, anchor_scales=args.anchor_scales,
                  num_head=args.num_head)

    model = model.to(args.device)

    model.apply(xavier_init)

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    max_val_fscore = -1

    train_set = VideoDataset(split['train_keys'])
    train_loader = DataLoader(train_set, shuffle=True)

    val_set = VideoDataset(split['test_keys'])
    val_loader = DataLoader(val_set, shuffle=False)

    for epoch in range(args.max_epoch):
        model.train()
        stats = AverageMeter('loss', 'cls_loss', 'loc_loss')

        for _, seq, gtscore, cps, n_frames, nfps, picks, _ in train_loader:
            keyshot_summ = VSummHelper.get_keyshot_summ(gtscore, cps, n_frames, nfps, picks)
            target = VSummHelper.downsample_summ(keyshot_summ)

            if not target.any():
                continue

            target_bboxes = BboxHelper.seq2bbox(target)
            target_bboxes = BboxHelper.lr2cw(target_bboxes)
            anchors = AnchorHelper.get_anchors(target.size, args.anchor_scales)
            # Get class and location label for positive samples
            cls_label, loc_label = AnchorHelper.get_pos_label(anchors, target_bboxes, args.pos_iou_thresh)

            # Get negative samples
            num_pos = cls_label.sum()
            cls_label_neg, _ = AnchorHelper.get_pos_label(anchors, target_bboxes, args.neg_iou_thresh)
            cls_label_neg = AnchorHelper.get_neg_label(cls_label_neg, int(args.neg_sample_ratio * num_pos))

            # Get incomplete samples
            cls_label_incomplete, _ = AnchorHelper.get_pos_label(anchors, target_bboxes, args.incomplete_iou_thresh)
            cls_label_incomplete[cls_label_neg != 1] = 1
            cls_label_incomplete = AnchorHelper.get_neg_label(cls_label_incomplete,int(args.incomplete_sample_ratio * num_pos))

            cls_label[cls_label_neg == -1] = -1
            cls_label[cls_label_incomplete == -1] = -1

            cls_label = torch.tensor(cls_label, dtype=torch.float32).to(args.device)
            loc_label = torch.tensor(loc_label, dtype=torch.float32).to(args.device)

            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(args.device)

            pred_cls, pred_loc = model(seq)

            loc_loss = calc_loc_loss(pred_loc, loc_label, cls_label)
            cls_loss = calc_cls_loss(pred_cls, cls_label)

            loss = cls_loss + args.lambda_reg * loc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stats.update(loss=loss.item(), cls_loss=cls_loss.item(),loc_loss=loc_loss.item())

        val_fscore = evaluate(model, val_loader, args.nms_thresh, args.device)

        if max_val_fscore < val_fscore:
            max_val_fscore = val_fscore
            torch.save(model.state_dict(), str(save_path))

        print(strftime("[ %Y-%m-%d %H:%M:%S ]", gmtime()),
                f'Epoch: {epoch}/{args.max_epoch} '
                f'Loss: {stats.cls_loss:.4f}/{stats.loc_loss:.4f}/{stats.loss:.4f} '
                f'F-score cur/max: {val_fscore:.4f}/{max_val_fscore:.4f}')

    return max_val_fscore


###--------------------------------------driver code--------------------------------------------------###

print("v36\n\n")

args = Parameter()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

model_dir = Path(args.model_dir)
model_dir.mkdir(parents=True, exist_ok=True)
DataHelper.get_ckpt_dir(model_dir).mkdir(parents=True, exist_ok=True)

DataHelper.dump_yaml(vars(args), model_dir / 'args.yml')

for split_path in args.splits:
    split_path = Path(split_path)
    splits = DataHelper.load_yaml(split_path)

    results = {}
    stats = AverageMeter('fscore')

    for split_idx, split in enumerate(splits):
        print('Start training on',split_path.stem,':',split , split_idx)
        ckpt_path = DataHelper.get_ckpt_path(model_dir, split_path, split_idx)
        fscore = trainer(args, split, ckpt_path)
        stats.update(fscore=fscore)
        results[f'split{split_idx}'] = float(fscore)

    results['mean'] = float(stats.fscore)
    DataHelper.dump_yaml(results, model_dir / f'{split_path.stem}.yml')

    print('Training done on ',split_path.stem,'. F-score: ','{:.4f}'.format(stats.fscore))
