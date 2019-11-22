classes= 80
train="/data/wanghy/coco/trainvalno5k.txt"
valid="/data/wanghy/coco/5k.txt"
names="data/coco.names"
backup="backup/"
eval="coco"


epochs=100
batch_size=16

model_def="config/yolov3.cfg"
data_config="config/coco.data"

pretrained_weights = "weights/darknet53.conv.74"

n_cpu=8
img_size=416

n_gpu='6,7'
base_lr=1e-3
weight_decay=0.000489