import os
import datetime
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm.autonotebook import tqdm
from pycocotools.cocoeval import COCOeval

from model import SSD, ResNet 
from dataloader import ObjectDataset
from utils import dboxes_generate, Encoder, make_dirs
from utils import SSDAugmentation, Loss
from utils import collate_fn
def train(modellib, train_loader, epoch, writer, loss_fn, optimizer, scheduler):
    modellib.train()
    num_iter_per_epoch = len(train_loader)
    progress_bar = tqdm(train_loader)
    for i, (img, _, _, grd_box, grd_lbl) in enumerate(progress_bar):
        if torch.cuda.is_available():
            img = img.cuda()
            grd_box = grd_box.cuda()
            grd_lbl = grd_lbl.cuda()
        modellib.eval()
        pred_box, pred_lbl = modellib(img)
        pred_box, pred_lbl = pred_box.float(), pred_lbl.float()
        # print("grd_box:", grd_box.shape)
        # print("pred_box:", pred_box.shape)
        # print("grd_lbl:", grd_lbl.shape)
        # print("pred_lbl:", pred_lbl.shape)
        grd_box = grd_box.transpose(1, 2).contiguous()
        loss = loss_fn(pred_box, pred_lbl, grd_box, grd_lbl)

        progress_bar.set_description("Epoch: {}. Loss: {:.5f}".format(epoch + 1, loss.item()))

        writer.add_scalar("Train Loss", loss.item(), epoch * num_iter_per_epoch + i)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


def evaluate(modellib, test_loader, epoch, writer, encoder, nms_threshold):
    modellib.eval()
    detections = []
    cat_ids = test_loader.dataset.coco.getCatIds()
    
    for nbatch, (img, img_id, img_size, _, _) in enumerate(test_loader):
        print("Parsing batch: {}/{}".format(nbatch, len(test_loader)), end="\r")
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            # Get predictions
            pred_box, pred_lbl = modellib(img)
            pred_box, pred_lbl = pred_box.float(), pred_lbl.float()

            for idx in range(pred_box.shape[0]):
                pred_box_i = pred_box[idx, :, :].unsqueeze(0)
                pred_lbl_i = pred_lbl[idx, :, :].unsqueeze(0)
                try:
                    result = encoder.decode_batch(pred_box_i, pred_lbl_i, nms_threshold, 200)[0]
                except:
                    print("No object detected in idx: {}".format(idx))
                    continue

                height, width = img_size[idx]
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    detections.append([img_id[idx], loc_[0] * width, loc_[1] * height, (loc_[2] - loc_[0]) * width,
                                       (loc_[3] - loc_[1]) * height, prob_,
                                       cat_ids[label_ - 1]])

    detections = np.array(detections, dtype=np.float32)
    coco_eval = COCOeval(test_loader.dataset.coco, test_loader.dataset.coco.loadRes(detections), iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    writer.add_scalar("Test mAP", coco_eval.stats[0], epoch)


def main():
    num_gpus = 1
    torch.cuda.manual_seed(123)
    batch_size = 1
    num_workers = 4
    log_dir = "logs"
    lr = 2.6e-3
    epochs = 10
    momentum = 0.9
    nms_threshold = 0.5
    weight_decay = 0.0005
    multi_step = [43, 54]
    data_path = "data/vehicles"
    save_folder = "trained_models"
    
    date_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    train_params = {"batch_size": batch_size,  "shuffle": True,
                    "drop_last": False, "num_workers": num_workers,
                    "collate_fn": collate_fn}

    test_params = {"batch_size": batch_size, "shuffle": False,
                   "drop_last": False, "num_workers": num_workers,
                   "collate_fn": collate_fn}

    dboxes = dboxes_generate()
    model = SSD(backbone=ResNet(), num_classes=4)
        
    train_set = ObjectDataset(data_path, "train", SSDAugmentation(dboxes, (300, 300), val=False))
    train_loader = DataLoader(train_set, **train_params)
    test_set = ObjectDataset(data_path, "val", SSDAugmentation(dboxes, (300, 300), val=True))
    test_loader = DataLoader(test_set, **test_params)

    encoder = Encoder(dboxes)

    lr = lr * (batch_size / 32)
    print(lr)
    loss_fn = Loss(dboxes)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay,
                                nesterov=True)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=multi_step, gamma=0.1)

    model.cuda()
    loss_fn.cuda()

    log_path = os.path.join(log_dir, date_time)
    save_path = os.path.join(log_path, save_folder)
    make_dirs(save_path)

    checkpoint_path = os.path.join(save_path, "model_ssd.pth")

    writer = SummaryWriter(log_path)

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        first_epoch = checkpoint["epoch"] + 1
        # model.module.load_state_dict(checkpoint["model_state_dict"])
        model.load_state_dict(checkpoint["model_state_dict"])

        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        first_epoch = 0

    for epoch in range(first_epoch, epochs):
        train(model, train_loader, epoch, writer, loss_fn, optimizer, scheduler)
        evaluate(model, test_loader, epoch, writer, encoder, nms_threshold)

        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(),
                      "optimizer": optimizer.state_dict(), 
                      "scheduler": scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    main()