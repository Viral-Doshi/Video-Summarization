import math
import numpy as np
import torch
import yaml
import h5py
from torch import nn
from pathlib import Path
from typing import List, Tuple, Iterable
from ortools.algorithms.pywrapknapsack_solver import KnapsackSolver

class DataHelper:
    def load_yaml(path):
        with open(path) as f:
            obj = yaml.safe_load(f)
        return obj

    def get_ckpt_path(model_dir, split_path, split_index):
        split_path = Path(split_path)
        return Path(model_dir) / 'checkpoint' / f'{split_path.name}.{split_index}.pt'

class AverageMeter(object):
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
        pred_bboxes = BboxHelper.cw2lr(pred_bboxes)

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

class VideoDataset(object):
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

class DataLoader(object):
    def __init__(self, dataset: VideoDataset, shuffle: bool):
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

class BboxHelper:
    def cw2lr(bbox_cw):
        bbox_cw = np.asarray(bbox_cw, dtype=np.float32).reshape((-1, 2))
        left = bbox_cw[:, 0] - bbox_cw[:, 1] / 2
        right = bbox_cw[:, 0] + bbox_cw[:, 1] / 2
        bbox_lr = np.vstack((left, right)).T
        return bbox_lr

    def iou_lr(anchor_bbox, target_bbox) -> np.ndarray:
        """Compute iou between multiple LR bbox pairs.

        :param anchor_bbox: LR anchor bboxes. Sized [N, 2].
        :param target_bbox: LR target bboxes. Sized [N, 2].
        :return: IoU between each bbox pair. Sized [N].
        """
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

    def nms(scores: np.ndarray, bboxes: np.ndarray, thresh: float) -> Tuple[np.ndarray, np.ndarray]:

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


    def get_summ_diversity(pred_summ, features):
        assert len(pred_summ) == len(features)
        pred_summ = np.asarray(pred_summ, dtype=np.bool)
        pos_features = features[pred_summ]

        if len(pos_features) < 2:
            return 0.0

        diversity = 0.0
        for feat in pos_features:
            diversity += (feat * pos_features).sum() - (feat * feat).sum()

        diversity /= len(pos_features) * (len(pos_features) - 1)
        return diversity


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


def evaluate(model, val_loader, nms_thresh, device):
    model.eval()
    stats = AverageMeter('fscore', 'diversity')

    with torch.no_grad():
        for test_key, seq, _, cps, n_frames, nfps, picks, user_summary in val_loader:
            seq_len = len(seq)
            seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)
            pred_cls, pred_bboxes = model.predict(seq_torch)
            pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)
            pred_cls, pred_bboxes = BboxHelper.nms(pred_cls, pred_bboxes, nms_thresh)

            pred_summ = VSummHelper.bbox2summary(seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks)

            eval_metric = 'avg' if 'tvsum' in test_key else 'max'
            fscore = VSummHelper.get_summ_f1score(pred_summ, user_summary, eval_metric)
            pred_summ = VSummHelper.downsample_summ(pred_summ)
            diversity = VSummHelper.get_summ_diversity(pred_summ, seq)
            stats.update(fscore=fscore, diversity=diversity)

    return stats.fscore, stats.diversity


args = Parameter()
model = DSNet(args.base_model, args.num_feature, args.num_hidden, args.anchor_scales, args.num_head)
model = model.eval().to(args.device)
# model = model.train(False).to(args.device)

for split_path in args.splits:
    split_path = Path(split_path)
    splits = DataHelper.load_yaml(split_path)

    stats = AverageMeter('fscore', 'diversity')
    for split_idx, split in enumerate(splits):
        ckpt_path = DataHelper.get_ckpt_path(args.model_dir, split_path, split_idx)
        state_dict = torch.load(str(ckpt_path),map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        val_set = VideoDataset(split['test_keys'])
        val_loader = DataLoader(val_set, shuffle=False)   

        fscore, diversity = evaluate(model, val_loader, args.nms_thresh, args.device)        
        stats.update(fscore=fscore, diversity=diversity)
        print(f'{split_path.stem} split {split_idx}: diversity: ' f'{diversity:.4f}, F-score: {fscore:.4f}')

    print(f'{split_path.stem}: diversity: {stats.diversity:.4f}, ' f'F-score: {stats.fscore:.4f}')