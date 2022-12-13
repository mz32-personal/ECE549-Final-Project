from classifier.dataset import GestureDataset
from classifier.preprocess import get_transform
from omegaconf import OmegaConf, DictConfig
import torch
import cv2
import matplotlib.pyplot as plt
import torchvision
from torchmetrics.functional import accuracy, f1_score, precision, recall
from tqdm import tqdm

path_to_config = "classifier/config/default.yaml"
conf = OmegaConf.load(path_to_config)
train_dataset = GestureDataset(is_train=True, conf=conf, transform=get_transform())
test_dataset = GestureDataset(is_train=False, conf=conf, transform=get_transform())

def collate_fn(batch): return tuple(zip(*batch)) # Collate func for dataloader


test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=conf.train_params.test_batch_size,
    num_workers=conf.train_params.num_workers,
    collate_fn=collate_fn,
    persistent_workers = False,
    prefetch_factor=conf.train_params.prefetch_factor,
)


model = torchvision.models.resnet18(num_classes = 19)
model.load_state_dict(torch.load(conf.model.checkpoint))
model.to(conf.device)

if __name__ == '__main__':
    # Testing
    all_acc = []
    with torch.no_grad():
        model.eval() # set model in eval mode
        for images, labels in tqdm(test_dataloader, desc= "testing"):
            # Process input tensor
            images = torch.stack(list(image.to(conf.device) for image in images))
            # Forward Propagation
            output = model(images)
            # Process ground truth labels
            target_labels = [label['gesture'] for label in labels]
            target_labels = torch.as_tensor(target_labels)
            target_labels = target_labels.to(conf.device)
            predicted = torch.argmax(output.detach(), dim=1)
            predicted = predicted.to(conf.device)
            # acc = (predicted == target_labels).sum() / len(predicted)
            acc = accuracy(predicted, target_labels, num_classes=19)
            all_acc.append(acc.item())
    print("Accuracy:", torch.mean(torch.tensor(all_acc)).item())