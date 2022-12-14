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

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=conf.train_params.train_batch_size,
    num_workers=conf.train_params.num_workers,
    collate_fn=collate_fn,
    persistent_workers = False,
    prefetch_factor=conf.train_params.prefetch_factor,
    shuffle=True
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=conf.train_params.test_batch_size,
    num_workers=conf.train_params.num_workers,
    collate_fn=collate_fn,
    persistent_workers = False,
    prefetch_factor=conf.train_params.prefetch_factor,
)


model = torchvision.models.resnet18(num_classes = 19)
model.to(conf.device)
params = [p for p in model.parameters() if p.requires_grad]
num_epochs = conf.train_params.epochs
optimizer = torch.optim.SGD(params, lr=conf.optimizer.lr, momentum=conf.optimizer.momentum, weight_decay=conf.optimizer.weight_decay)

if __name__ == '__main__':
    # Training:
    print("starting training")
    model.train() # set model in training mode
    criterion = torch.nn.CrossEntropyLoss()
    loss_per_epoch = []
    for epoch in tqdm(range(num_epochs), desc = "Training"):
        for images, labels in train_dataloader:
            # Process input tensor
            images = torch.stack(list(image.to(conf.device) for image in images))
            # Forward Propagation
            output = model(images)
            # Process ground truth labels
            target_labels = [label['gesture'] for label in labels]
            target_labels = torch.as_tensor(target_labels).to(conf.device)
            # calculate loss function
            loss = criterion(output, target_labels)
            loss_per_epoch.append(loss.item())
            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # save model
    torch.save(model.state_dict(), conf.model.checkpoint)