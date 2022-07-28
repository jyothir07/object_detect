import torch
from torchvision.datasets import CocoDetection
import os
from torch.utils.data.dataloader import default_collate

def collate_fn(batch):
    datas = list(zip(*batch))
    datas[0] = default_collate([i for i in datas[0] if torch.is_tensor(i)])
    datas[1] = list([i for i in datas[1] if i])
    datas[2] = list([i for i in datas[2] if i])
    datas[3] = default_collate([i for i in datas[3] if torch.is_tensor(i)])
    datas[4] = default_collate([i for i in datas[4] if torch.is_tensor(i)])
    return datas

class ObjectDataset(CocoDetection):

    def __init__(self, root, mode, transform=None):
        annot_file = os.path.join(root, "labels", "annotations_{}.json".format(mode))
        root = os.path.join(root, "{}".format(mode))
        super(ObjectDataset, self).__init__(root, annot_file)
        self._load_categories()
        self.transform = transform

    def _load_categories(self):

        cats = self.coco.loadCats(self.coco.getCatIds())
        
        cats.sort(key=lambda x: x["id"])

        self.lbl_map = {}
        self.lbl_info = {}
        count = 1
        self.lbl_info[0] = "background"
        for c in cats:
            self.lbl_map[c["id"]] = count
            self.lbl_info[count] = c["name"]
            count += 1

    def __getitem__(self, item):
        image, result = super(ObjectDataset, self).__getitem__(item)
        width, height = image.size
        boxes = []
        lbls = []
        if len(result) == 0:
            return None, None, None, None, None
        for annotation in result:
            bbox = annotation.get("bbox")
            boxes.append([bbox[0] / width, bbox[1] / height, (bbox[0] + bbox[2]) / width, (bbox[1] + bbox[3]) / height])
            lbls.append(self.lbl_map[annotation.get("category_id")])
        boxes = torch.tensor(boxes)
        lbls = torch.tensor(lbls)
        if self.transform is not None:
            image, (height, width), boxes, lbls = self.transform(image, (height, width), boxes, lbls)
        return image, result[0]["image_id"], (height, width), boxes, lbls