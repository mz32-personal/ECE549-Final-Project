from classifier.dataset import GestureDataset
from classifier.preprocess import get_transform
from omegaconf import OmegaConf, DictConfig
import torch

path_to_config = "classifier/config/default.yaml"
conf = OmegaConf.load(path_to_config)
train_dataset = GestureDataset(is_train=True, conf=conf, transform=get_transform())
test_dataset = GestureDataset(is_train=False, conf=conf, transform=get_transform())

collate_fn = lambda batch: tuple(zip(*batch)) # Collate func for dataloader

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=conf.train_params.train_batch_size,
    num_workers=conf.train_params.num_workers,
    collate_fn=collate_fn,
    persistent_workers = True,
    prefetch_factor=conf.train_params.prefetch_factor,
    shuffle=True
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=conf.train_params.test_batch_size,
    num_workers=conf.train_params.num_workers,
    collate_fn=collate_fn,
    persistent_workers = True,
    prefetch_factor=conf.train_params.prefetch_factor,
)

# TODO: Make sense of how the dataset works

# TODO: define a neural network model

# TODO: set up an optimizer and loss funciton (criterion)

# TODO: set up training loop and back propagation (loss.backward())

# TODO: test model accuracy