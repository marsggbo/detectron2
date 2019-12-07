import torch
import numpy as np

def create_maskimg(x, boxes, theta=0.95, save_maskimg=False):
    '''
    params:
        x: tensor (bs, c, h, w).
        boxes: list. Each item indicates the infomation of 
    '''
    count = 0
    samples = []
    _, _, h, w = x.shape
    img = x.clone()
    for i, box in enumerate(boxes):
        mask = torch.zeros_like(x[i]) # 3*h*w
        for b in box:
            x1, y1, x2, y2 = b
            if (x2-x1)*(y2-y1) < 100:
                scale1 = 0.5 if x2-x1 < 20 else 0.8
                scale2 = 1.5 if y2-y1 < 20 else 1.2
                x1 *= scale1 
                y1 *= scale1
                x2 *= scale2 if scale2*x2 < w else w
                y2 *= scale2 if scale2*y2 < h else h
            cx, cy = int((x2-x1)/2.), int((y2-y1)/2.)
            x1, y1, x2, y2 = list(map(lambda x:int(x), [x1, y1, x2, y2]))

            def distance(c, i, j):
                d = ((i-cx)**2+(j-cy)**2)**0.5
                return theta**d
            tmp_mask = np.fromfunction(distance, (3, h, w))
            mask += torch.from_numpy(tmp_mask).float().cuda()
        mask /= len(box)
        mask[:,x1:x2,y1:y2] = 1
        img[i] *= mask
        if save_maskimg:
            if count <= 6:
                samples.append(img[i].cpu().numpy())
                count += 1
            else:
                samples = np.stack(samples)
                np.save(f"{count}.npy", samples)
    return x

def topk_acc(preds, gt_classes, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)

    _, pred = preds.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(gt_classes.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k)
        # res.append(correct_k.mul_(100.0 / batch_size))
    return res