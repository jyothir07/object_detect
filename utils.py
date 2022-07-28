import numpy as np
import itertools
from math import sqrt
import os
import torch
import torch.nn.functional as F
import torch.nn as nn

from torchvision.ops.boxes import box_convert, box_iou
import torchvision.transforms as transforms

import random
from PIL import Image

class Loss(nn.Module):
    """
        the sum of conf Loss and localization Loss
    """

    def __init__(self, det_box):
        super(Loss, self).__init__()
        self.scale_xy = 1.0 / det_box.scale_xy
        self.scale_wh = 1.0 / det_box.scale_wh

        self.smoothl1_loss = nn.SmoothL1Loss(reduce=False)
        self.det_box = nn.Parameter(det_box(order="xywh").transpose(0, 1).unsqueeze(dim=0), requires_grad=False)
        self.ce_loss = nn.CrossEntropyLoss(reduce=False)

    def loc_vec(self, loc):
        gxy = self.scale_xy * (loc[:, :2, :] - self.det_box[:, :2, :]) / self.det_box[:, 2:, ]
        gwh = self.scale_wh * (loc[:, 2:, :] / self.det_box[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, pred_box, pred_lbl, grd_box, grd_lbl):
        """
            pred_box, pred_lbl: Nx4x8732, Nxlabel_numx8732
                predicted boxes and labels

            grd_box, grd_lbl: Nx4x8732, Nx8732
                ground truth boxes and labels
        """
        mask = grd_lbl > 0
        pos_num = mask.sum(dim=1)

        # vec_gd = self.loc_vec(grd_box)

        # sum on four coordinates, and mask
        smooth_l1 = self.smoothl1_loss(pred_box, self.loc_vec(grd_box)).sum(dim=1)
        smooth_l1 = (mask.float() * smooth_l1).sum(dim=1)

        # hard negative mining
        con = self.ce_loss(pred_lbl, grd_lbl)

        # postive mask will never selected
        con_neg = con.clone()
        con_neg[mask] = 0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)

        # number of negative three times positive
        neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num

        closs = (con * (mask.float() + neg_mask.float())).sum(dim=1)
        # closs = con * (mask.float()).sum(dim=1)
        # avoid no object detected
        total_loss = smooth_l1 + closs
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss * num_mask / pos_num).mean(dim=0)
        return ret

class BoxesGen(object):
    def __init__(self, scale_xy=0.1, scale_wh=0.2):
        
        self.scale_xy = scale_xy
        self.scale_wh = scale_wh
        self.fig_sz = 300
        self.featr_sz = [38, 19, 10, 5, 3, 1]
        self.steps = [8, 16, 32, 64, 100, 300]
        self.scales = [21, 45, 99, 153, 207, 261, 315]
        self.ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

        fk = self.fig_sz / np.array(self.steps)

        self.default_boxes = []
        for idx, sfeat in enumerate(self.featr_sz):

            sk1 = self.scales[idx] / self.fig_sz
            sk2 = self.scales[idx + 1] / self.fig_sz
            sk3 = sqrt(sk1 * sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in self.ratios[idx]:
                w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.det_box = torch.tensor(self.default_boxes, dtype=torch.float)
        self.det_box.clamp_(min=0, max=1)
        self.det_box_ltrb = box_convert(self.det_box, in_fmt="cxcywh", out_fmt="xyxy")

    def __call__(self, order="ltrb"):
        if order == "ltrb":
            return self.det_box_ltrb
        else:  # order == "xywh"
            return self.det_box
        
class Encoder(object):

    def __init__(self, det_box):
        self.det_box = det_box(order="ltrb")
        self.det_box_xywh = det_box(order="xywh").unsqueeze(dim=0)
        self.nboxes = self.det_box.size(0)
        self.scale_xy = det_box.scale_xy
        self.scale_wh = det_box.scale_wh

    def encode(self, bbxs_in, labels_in, criteria=0.5):
        # print("bbox:", bbxs_in, labels_in)
        ious = box_iou(bbxs_in, self.det_box)
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)

        # set best ious 2.0
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)

        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
        best_dbox_idx[best_bbox_idx[idx]] = idx

        # filter IoU > 0.5
        masks = best_dbox_ious > criteria
        labels_out = torch.zeros(self.nboxes, dtype=torch.long)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]
        bbxs_out = self.det_box.clone()
        bbxs_out[masks, :] = bbxs_in[best_dbox_idx[masks], :]
        bbxs_out = box_convert(bbxs_out, in_fmt="xyxy", out_fmt="cxcywh")
        return bbxs_out, labels_out

    def scale_back_batch(self, bbxs_in, scores_in):
        """
            from xywh to ltrb, input Nx4xnum_bbox Nxlabel_numxnum_bbox
        """
        if bbxs_in.device == torch.device("cpu"):
            self.det_box = self.det_box.cpu()
            self.det_box_xywh = self.det_box_xywh.cpu()
        else:
            self.det_box = self.det_box.cuda()
            self.det_box_xywh = self.det_box_xywh.cuda()

        bbxs_in = bbxs_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)

        bbxs_in[:, :, :2] = self.scale_xy * bbxs_in[:, :, :2]
        bbxs_in[:, :, 2:] = self.scale_wh * bbxs_in[:, :, 2:]

        bbxs_in[:, :, :2] = bbxs_in[:, :, :2] * self.det_box_xywh[:, :, 2:] + self.det_box_xywh[:, :, :2]
        bbxs_in[:, :, 2:] = bbxs_in[:, :, 2:].exp() * self.det_box_xywh[:, :, 2:]
        bbxs_in = box_convert(bbxs_in, in_fmt="cxcywh", out_fmt="xyxy")

        return bbxs_in, F.softmax(scores_in, dim=-1)

    def decode_batch(self, bbxs_in, scores_in, nms_threshold=0.45, max_output=200):
        bboxes, probs = self.scale_back_batch(bbxs_in, scores_in)
        output = []
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            output.append(self.decode_single(bbox, prob, nms_threshold, max_output))
        return output

    def decode_single(self, bbxs_in, scores_in, nms_threshold, max_output, max_num=200):
        bbxs_out = []
        scores_out = []
        labels_out = []

        for i, bbx_score in enumerate(scores_in.split(1, 1)):
            if i == 0:
                continue

            bbx_score = bbx_score.squeeze(1)
            mask = bbx_score > 0.05

            bboxes, bbx_score = bbxs_in[mask, :], bbx_score[mask]
            if bbx_score.size(0) == 0: continue

            score_sorted, score_idx_sorted = bbx_score.sort(dim=0)

            # select max_output indices
            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = []

            while score_idx_sorted.numel() > 0:
                idx = score_idx_sorted[-1].item()
                bboxes_sorted = bboxes[score_idx_sorted, :]
                bboxes_idx = bboxes[idx, :].unsqueeze(dim=0)
                iou_sorted = box_iou(bboxes_sorted, bboxes_idx).squeeze()
                # we only need iou < nms_threshold
                score_idx_sorted = score_idx_sorted[iou_sorted < nms_threshold]
                candidates.append(idx)

            bbxs_out.append(bboxes[candidates, :])
            scores_out.append(bbx_score[candidates])
            labels_out.extend([i] * len(candidates))

        if not bbxs_out:
            return [torch.tensor([]) for _ in range(3)]

        bbxs_out, labels_out, scores_out = torch.cat(bbxs_out, dim=0), \
                                             torch.tensor(labels_out, dtype=torch.long), \
                                             torch.cat(scores_out, dim=0)

        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]
        return bbxs_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, bboxes):
        if random.random() < self.prob:
            bboxes[:, 0], bboxes[:, 2] = 1.0 - bboxes[:, 2], 1.0 - bboxes[:, 0]
            return img.transpose(Image.FLIP_LEFT_RIGHT), bboxes
        return img, bboxes

class SSDAugmentation(object):
    def __init__(self, det_box, size=(300, 300), val=False):
        self.size = size
        self.val = val
        self.det_box = det_box
        self.encoder = Encoder(self.det_box)

        self.hflip = RandomHorizontalFlip()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.img_trans = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
            transforms.ToTensor(),
            self.normalize
        ])
        self.trans_val = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            self.normalize
        ])

    def __call__(self, img, img_size, bboxes=None, labels=None, max_num=200):
        if self.val:
            bbox_out = torch.zeros(max_num, 4)
            label_out = torch.zeros(max_num, dtype=torch.long)
            bbox_out[:bboxes.size(0), :] = bboxes
            label_out[:labels.size(0)] = labels
            return self.trans_val(img), img_size, bbox_out, label_out

        img, bboxes = self.hflip(img, bboxes)

        img = self.img_trans(img).contiguous()
        bboxes, labels = self.encoder.encode(bboxes, labels)

        return img, img_size, bboxes, labels
