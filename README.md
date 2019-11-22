# PyTorch-YOLOv3-Multi_GPUs-Training
Training [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) with multi-GPUs.
## Installation
##### Clone and install requirements
    $ git clone https://github.com/eriklindernoren/PyTorch-YOLOv3
    $ cd PyTorch-YOLOv3/
    $ sudo pip3 install -r requirements.txt

##### Download pretrained weights
    $ cd weights/
    $ bash download_weights.sh

##### Download COCO
    $ cd data/
    $ bash get_coco_dataset.sh
    
## Train with COCO
To train on COCO using a Darknet-53 backend pretrained on ImageNet run: 
```
$ Modify the configs in config/config.py, especially paths of data and weights.
$ python train.py 
```

## What we modify

We make the same number of targets for each image in function **ListDataset.collate_fn** of utils/dataset.py.
