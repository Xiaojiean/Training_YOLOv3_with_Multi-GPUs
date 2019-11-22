from __future__ import division

import warnings
warnings.filterwarnings("ignore")

from models import *
#from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
import config.config as cfg
from torch.optim import lr_scheduler
from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    with torch.no_grad():
        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale target
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= img_size
            imgs = imgs.type(Tensor)
            outputs = model(imgs)
            outputs = to_cpu(outputs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=cfg.epochs, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default=cfg.model_def, help="path to model definition file")
    parser.add_argument("--pretrained_weights", type=str, default=cfg.pretrained_weights,help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=cfg.n_cpu, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=cfg.img_size, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--n_gpu",type=str, default=cfg.n_gpu, help="allow for multi-scale training")
    parser.add_argument('--base_lr', type=float, default=cfg.base_lr)
    parser.add_argument('--weight_decay', type=float, default=cfg.weight_decay)
    opt = parser.parse_args()
    print(opt)
    #logger = Logger("logs")

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.n_gpu
    print(torch.cuda.device_count())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    train_path = cfg.train
    valid_path = cfg.valid
    class_names = load_classes(cfg.names)



    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)



    model = nn.DataParallel(model)


    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.base_lr, weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(cfg.epochs * x) for x in [0.8, 0.9]], gamma=0.1)

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]
    warm_up_batchs = cfg.warmup
    wp_batch_cnt = 0
    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        end_train = 0

        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            if batch_i == len(dataloader)-1:
                break
            batches_done = len(dataloader) * epoch + batch_i
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            loss, outputs = model(imgs, targets)
            loss.mean().backward()

            if int(wp_batch_cnt / opt.gradient_accumulations) < warm_up_batchs:
                wp_batch_cnt += 1
                alpha = float(wp_batch_cnt) / warm_up_batchs
                warmup_factor = 1./3 * (1 - alpha) + alpha
                lr=cfg.base_lr*warmup_factor
                if  (wp_batch_cnt+1)%10==0:
                    print('warm up lr: ')
                    print(lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            if (batch_i+1)%100 == 0:
                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

                metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.module.yolo_layers))]]]

                #Log metrics at each YOLO layer
                for i, metric in enumerate(metrics):
                    formats = {m: "%.6f" for m in metrics}
                    formats["grid_size"] = "%2d"
                    formats["cls_acc"] = "%.2f%%"
                    row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.module.yolo_layers]
                    metric_table += [[metric, *row_metrics]]

                    # Tensorboard logging
                    tensorboard_log = []

                    for j, yolo in enumerate(model.module.yolo_layers):
                        #print(yolo.metrics.items())
                        for name, metric in yolo.metrics.items():
                            if name != "grid_size":
                                tensorboard_log += [(f"{name}_{j+1}", metric)]
                    tensorboard_log += [("loss", loss.mean().item())]
                    #logger.list_of_scalars_summary(tensorboard_log, batches_done)

                log_str += AsciiTable(metric_table).table
                log_str += f"\nTotal loss {loss.mean().item()}"

                # Determine approximate time left for epoch
                epoch_batches_left = len(dataloader) - (batch_i + 1)
                time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                log_str += f"\n---- ETA {time_left}"
                print(log_str)

            model.module.seen += imgs.size(0)
        if wp_batch_cnt>warm_up_batchs:
            scheduler.step()
        #print('Epoch:{}, training time (m): {}, validation time (m): {}'.format(epoch, (end_train-start_time)/60, 0))
        end_train = time.time()
        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.01,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=20,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            #logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
        end_val=time.time()
        print('time statistics')
        print('Epoch:{}, training time (m): {}, validation time (m): {}'.format(epoch, (end_train-start_time)/60, (end_val-end_train)/60))
        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)


