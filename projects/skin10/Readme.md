
# 数据集格式
首先将数据集放到skin路径下，格式如下：

```bash
path/to/detectron2/projects/skin10/datasets/coco
```

coco文件夹格式如下：

- annotations(存放json文件)
  - instances_train2017.json
  - instances_test2017.json
  - instances_val2017.json
- test2017(存放图片，不需要分文件夹，因为在json中已经将图片的信息记录下来了，下面两个文件夹同理)
- train2017
- val2017


# 训练
```bash
python projects/skin10/skin_train_net.py --config-file projects/skin10/config/faster_rcnn_R_101_FPN_3x.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

训练后会在`skin10/output`路径下生成一系列的`pth`,`checkpoint`和`log.txt`等文件

# 评估

```bash
python projects/skin10/skin_train_net.py --config-file projects/skin10/config/faster_rcnn_R_101_FPN_3x.yaml --eval-only MODEL.WEIGHTS output/model_0219999.pth
```

评估后会在`skin10/output`路径下生成`inference`文件夹，包含如下两个文件：
- coco_instances_results.json  
- instances_predictions.pth


# 版本修改信息

## 2019.12.10

- 修改训练过程中弄存在的bug
  - `roi_img_head.py`：使用resnet的预训练模型
- 正常运行inference
  - 数据：修改测试集数据读入的处理方式
    - `skin_train_net.py`: 修改Trainer中`build_test_loader`函数
    - `test_dataset_mapper.py`: 用于指定测试集的数据处理方式，保留了`instances`信息，在最后计算图像分类准确率提供gt_classes
  - 模型：
    - `rcnn`: 
      - 添加了一些注解
      - inference步骤中结果后处理做了修改，将box预测和图片分类预测结果合并
  - 结果：
    - `img_acc_evaluator.py`
      - 修改process和evaluate函数中的逻辑bug
- Todo
  - [ ] 测试多GPU训练