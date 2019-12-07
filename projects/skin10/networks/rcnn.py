
import logging
import torch
from torch import nn

from detectron2.modeling import GeneralizedRCNN, detector_postprocess
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.modeling import META_ARCH_REGISTRY
from .roi_img_head import build_roi_img_head
 

# @META_ARCH_REGISTRY.register()
class SkinRCNN(GeneralizedRCNN):

    def __init__(self, cfg):
        super(SkinRCNN, self).__init__(cfg)
        self.roi_img_head = build_roi_img_head(cfg, input_shape=None)
        self.num_boxes = cfg.MODEL.SOFT_MASK.NUM
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # backbone
        features = self.backbone(images.tensor)

        # rpn
        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # roi_boxes
        proposals, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        # roi_imgs
        gt_classes = []
        for x in batched_inputs:
            gt_classes.append(x['instances'].gt_classes[0])
        gt_classes = torch.tensor(gt_classes).to(self.device)

        softmasks = []
        for p in proposals:
            boxes = p.proposal_boxes[:self.num_boxes]
            softmasks.append(boxes)
        
        _, img_cls_losses =  self.roi_img_head(images.tensor, gt_classes, softmasks)


        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(img_cls_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        gt_classes = []
        for x in batched_inputs:
            gt_classes.append(x['instances'].gt_classes[0])
        gt_classes = torch.tensor(gt_classes).to(self.device)

        softmasks = []
        for p in proposals:
            boxes = p.proposal_boxes[:self.num_boxes]
            softmasks.append(boxes)
        
        img_preds =  self.roi_img_head(images.tensor, gt_classes, softmasks)
        results.update(img_preds)

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results
