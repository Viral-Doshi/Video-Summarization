import logging
from pathlib import Path

import numpy as np
import torch

from anchor_based.dsnet import DSNet
from anchor_free.dsnet_af import DSNetAF
from helpers import data_helper, vsumm_helper, bbox_helper
from modules.model_zoo import get_model

logger = logging.getLogger()


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

    print(stats.fscore, stats.diversity)
    return stats.fscore, stats.diversity

# python evaluate.py anchor-based --model-dir ../models/pretrain_ab_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml

class Parameter:
    def __init__(self):
        
        self.model = 'anchor-based'
        self.device = "cuda"
        self.seed = 12345
        self.splits = ["../splits/tvsum.yml", "../splits/summe.yml"]
        self.max_epoch = 300
        self.model_dir = ["../models/pretrain_ab_basic/"][0]
        self.log_file = "log.txt"
        self.lr = 5e-5
        self.weight_decay = 1e-5
        self.lambda_reg = 1.0
        self.nms_thresh = 0.5

        self.ckpt_path = None
        self.sample_rate = 15
        self.source = None
        self.save_path = None

        self.base_model = ['attention', 'lstm', 'linear', 'bilstm', 'gcn'][0]
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
        self.cls_loss = ['focal', 'cross-entropy'][0]
        self.reg_loss = ['soft-iou', 'smooth-l1'][0]


    def get_model(self):
        if self.model == 'anchor-based':
            return DSNet(self.base_model, self.num_feature, self.num_hidden, self.anchor_scales, self.num_head)
        elif self.model == 'anchor-free':
            return DSNetAF(self.base_model, self.num_feature, self.num_hidden, self.num_head)
        else:
            print(f'Invalid model type {self.model_type}')





def main():
    args = Parameter()
    # args = init_helper.get_arguments()
    # init_helper.init_logger(args.model_dir, args.log_file)
    # init_helper.set_random_seed(args.seed)
    # logger.info(vars(args))

    model = args.get_model()
    model = model.eval().to(args.device)

    for split_path in args.splits:
        split_path = Path(split_path)
        splits = data_helper.load_yaml(split_path)

        stats = data_helper.AverageMeter('fscore', 'diversity')

        for split_idx, split in enumerate(splits):
            ckpt_path = data_helper.get_ckpt_path(args.model_dir, split_path, split_idx)
            state_dict = torch.load(str(ckpt_path),
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)

            val_set = data_helper.VideoDataset(split['test_keys'])
            val_loader = data_helper.DataLoader(val_set, shuffle=False)

            fscore, diversity = evaluate(model, val_loader, args.nms_thresh, args.device)
            stats.update(fscore=fscore, diversity=diversity)

            logger.info(f'{split_path.stem} split {split_idx}: diversity: '
                        f'{diversity:.4f}, F-score: {fscore:.4f}')

        logger.info(f'{split_path.stem}: diversity: {stats.diversity:.4f}, '
                    f'F-score: {stats.fscore:.4f}')


if __name__ == '__main__':
    main()
