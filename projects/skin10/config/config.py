from detectron2.config import CfgNode as CN

def add_skinrcnn_config(cfg):
    _C = cfg

    
    _C.MODEL.ROI_IMG_HEAD = CN()
    _C.MODEL.ROI_IMG_HEAD.NAME = "ROI_IMG_HEAD"
    _C.MODEL.ROI_IMG_HEAD.TYPE = "resnet50"

    _C.MODEL.SOFT_MASK = CN()
    _C.MODEL.SOFT_MASK.NUM = 5
    _C.MODEL.SOFT_MASK.THETA = 0.95

    _C.OUTPUT = CN()
    _C.OUTPUT.TOPK_ACC = 3