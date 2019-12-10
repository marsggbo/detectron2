import copy
import itertools
import logging
import os
from collections import OrderedDict
import torch
from fvcore.common.file_io import PathManager

import detectron2.utils.comm as comm
from detectron2.evaluation import DatasetEvaluator
from .utils import topk_acc


class ImgAccEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self._distributed = distributed
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._topk_acc = cfg.OUTPUT.TOPK_ACC
    
    def reset(self):
        self._gt_classes = []
        self._pred_classes = []
        self._acc = 0.
    
    def process(self, inputs, outputs):

        for input, output in zip(inputs, outputs):

            if 'img_cls_pred' in output:
                self._gt_classes.append(input['instances'].gt_classes[0])
                img_cls_pred = output['img_cls_pred'].to(self._cpu_device)
                self._pred_classes.append(img_cls_pred)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._pred_classes = comm.gather(self._pred_classes, dst=0)
            self._pred_classes = list(itertools.chain(*self._pred_classes))

            if not comm.is_main_process():
                return {}

        if len(self._pred_classes) == 0:
            self._logger.warning("[ImgClsEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "imgCls_gt_pred.pth")
            acc_results = topk_acc(torch.stack(self._pred_classes), torch.tensor(self._gt_classes), 
                    (1, self._topk_acc))
            
            imgCls_gt_pred = {
                'gt_classes': torch.tensor(self._gt_classes),
                'pred_classes': torch.stack(self._pred_classes),
                'top1_acc': acc_results[0],
                f"top{self._topk_acc}_acc": acc_results[1]
            }
            self._logger.info("Saving img classification results to {}".format(file_path))
            with PathManager.open(file_path, "wb") as f:
                torch.save(imgCls_gt_pred, f)

        self._results = OrderedDict()
        self._results['img_cls'] = {
            'top1_acc': acc_results[0],
            f'top{self._topk_acc}_acc': acc_results[1]
        }
        self._logger.info(f"top1_acc={acc_results[0]:.2f} top{self._topk_acc}_acc={acc_results[1]:.2f}")
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)
