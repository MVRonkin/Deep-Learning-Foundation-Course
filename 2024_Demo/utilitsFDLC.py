import os
import time
import random
import platform
import requests
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.utils.data 
import torch.nn.functional as F

import torchvision 
from torchvision import transforms, datasets
from torchvision.datasets.utils import download_and_extract_archive

from pprint import pprint
from torchinfo import summary
from tqdm.notebook import tqdm, trange
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import timm
from transformers import MobileViTForImageClassification

import lightning as pl
from lightning.pytorch.loggers import CSVLogger
from torchmetrics.classification import  Accuracy

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def set_root():
    if platform.system().startswith(('Win','win')):
        path = Path(os.getcwd())
        root_directory = os.path.join(path.anchor, 'data')
    else:
        home = str(Path.home())
        root_directory = os.path.join(home, 'data')
    return Path(root_directory)
    
#-----------------------------------
def torch_set_device(): 
    print('torch version:',".".join(torch.__version__.split(".")[:2]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device.type == 'cuda':
        print("cuda: ", torch.__version__.split("+")[-1])
        print('number of devices: %d'%(torch.cuda.device_count()))
    num_workers=os.cpu_count()
    print ('available number of workers:',num_workers)
    
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            device = torch.device("mps")
            print("MPS device")    
            
    return device, num_workers
    
#-----------------------------------
def freeze_model(model):
    for name,p in model.named_parameters():
        if not name.startswith(('classifier','fc','head')):
            p.requires_grad = False
    return model
    
#-----------------------------------
def replace_last_linear_layer(model, n_classes):
    
    *_,last = model.named_children()
    out = filter(lambda x: isinstance(x,nn.Linear), model.get_submodule(last[0]).modules())
    if len(list(out)) <1:
        raise ValueError( f"Unexpected architecture of {last[0]},\n{last[1]}")
    return model   

    def init_linear(layer):
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0)
        return layer

    if last[0] == 'classifier':
        if isinstance(last[1], nn.Sequential):
            n_fc_in = model.classifier[-1].in_features # HEAD, CLASSIFIER, FC
            model.classifier[-1] = init_linear(nn.Linear(n_fc_in, n_classes))
        else:
            n_fc_in = model.classifier.in_features # HEAD, CLASSIFIER, FC
            model.classifier = init_linear(nn.Linear(n_fc_in, n_classes))
    elif last[0] == 'fc':
        n_fc_in = model.fc.in_features
        model.fc = init_linear(nn.Linear(n_fc_in, n_classes))
    elif last[0] == 'head':
        if isinstance(last[1], nn.Sequential):
            n_fc_in = model.head[-1].in_features # HEAD, CLASSIFIER, FC
            model.head[-1] = init_linear(nn.Linear(n_fc_in, n_classes))    
        else:
            n_fc_in = model.head.in_features # HEAD, CLASSIFIER, FC
            model.head = init_linear(nn.Linear(n_fc_in, n_classes))     
    else:
       raise ValueError( f"Unexpected architecture of {last[0]},\n{last[1]}")
    
    return model
    
#-----------------------------------
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

#-----------------------------------
def torch_imshow(inp, title=None, figsize=(28,18), mean = MEAN, std = STD):
    """Imshow for Tensor."""
    inp = inp.data.cpu().numpy().transpose((1, 2, 0))
    inp = std * inp + mean 
    inp = np.clip(inp, 0, 1)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(inp, interpolation='nearest')
    
    if title is not None:
        plt.title(title)
        
    ax.set_axis_off()
    plt.show()
    
#-----------------------------------
def plot_confusion_matrix(labels, pred_labels, names_classes, figsize=(18, 10)):
    
    fig, ax = plt.subplots(figsize=figsize)
    
    cm = confusion_matrix(np.asarray(labels, dtype=int), 
                          np.asarray(pred_labels, dtype=int) )
    
    cm = ConfusionMatrixDisplay(cm, display_labels=names_classes)
    
    cm.plot(values_format='d', cmap='coolwarm', ax=ax)

#-----------------------------------
def eval_macro(model, data_loader, n_classes):
    
    predicts, labels, probs = [], [], []

    model.eval()
    with torch.inference_mode():
        for x, y  in tqdm(data_loader, desc="eval_macro", leave=False):
            output = model(x)
            cls_pred = torch.argmax(output, dim=1)
            y_prob = F.softmax(output, dim=-1)
            predicts.append(cls_pred), labels.append(y), probs.append(y_prob)
                
    labels   = torch.cat(labels, dim = 0) 
    predicts = torch.cat(predicts, dim = 0) # make it 1d array
    probs    = torch.cat(probs, dim=0)

    return labels.cpu(), predicts.cpu(), probs.cpu()

#-----------------------------------
def eval_micro(labels, predicts, n_classes):
    class_correct = torch.zeros(n_classes)
    class_total   = torch.zeros(n_classes)    
    for idx, (yi, y_hat) in enumerate(zip(labels,predicts)):
        if y_hat == yi:
            class_correct[yi] += 1            
        class_total[yi]   += 1
    return class_correct, class_total

#-----------------------------------
def eval_incorrect(labels, predicts, probs, data):
    incorrect_examples = []
    corrects = torch.eq(labels, predicts)
    for image, label, prob, correct in tqdm(zip(data, labels, probs, corrects), desc="eval_incorrect", leave=False):
        if not correct:
            incorrect_examples.append((image, label, prob))    

    incorrect_examples.sort(reverse=True,
                            key=lambda x: torch.max(x[2], dim=0).values)
    return incorrect_examples

#-----------------------------------
def plot_most_incorrect(incorrect, n_images, mean = MEAN, std = STD):

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(20, 20))
    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)
        image, true_label, probs = incorrect[i]
        true_prob = probs[true_label]
        incorrect_prob, incorrect_label = torch.max(probs, dim=0)

        img = image[0].cpu().numpy().transpose((1,2,0))
        img = std * img + mean 
        img = np.clip(img, 0, 1)

        ax.imshow(img, cmap='bone')
        ax.set_title(f'true label: {true_label} ({true_prob:.3f})\n'
                     f'pred label: {incorrect_label} ({incorrect_prob:.3f})')
        ax.axis('off')
    fig.subplots_adjust(hspace=0.5)

#-----------------------------------
class ModelPL(pl.LightningModule):
    def __init__(self, lr=0.01, n_classes = 9):
        super().__init__()

        self.lr = lr
        self.n_classes = n_classes

        self.set_model(model)

        self.criterion = nn.CrossEntropyLoss()

        self.val_accuracy   = Accuracy(task="multiclass", num_classes=self.n_classes)
        self.test_accuracy  = Accuracy(task="multiclass", num_classes=self.n_classes)
        self.save_hyperparameters(ignore=['model'])

    def set_model(self, torch_model  = None):
        self.model = torch_model
        return self

    def forward(self, x):
        x = self.model(x)
        return x

    def any_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return preds, loss

    def training_step(self, batch, batch_idx):
        _, loss = self.any_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds, loss = self.any_step(batch, batch_idx)

        self.val_accuracy.update(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds, loss = self.any_step(batch, batch_idx)

        self.val_accuracy.update(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.val_accuracy, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        return opt

#-----------------------------------
def clean_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    to_remove = set()
    for k, v in globals().items():
        if isinstance(v, (torch.nn.Module)):
            to_remove.add(k)
    for k in to_remove:
        del globals()[k]
    gc.collect()

#-----------------------------------
def test_cuda():
    torch_version = ".".join(torch.__version__.split(".")[:2])
    print('torch version:',torch_version)
    
    if device.type == 'cuda':
        cuda_version  = torch.__version__.split("+")[-1]
        print("cuda: ", cuda_version)
    
        n_devices = torch.cuda.device_count()
        print('number of devices: %d'%(n_devices))
    
        for cnt_device in range(n_devices):
            print(torch.cuda.get_device_name(cnt_device))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(cnt_device)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(cnt_device)/1024**3,1), 'GB')

#-----------------------------------
def get_squeezenet_1_0(n_classes ):
    model = torchvision.models.squeezenet1_0(weights=torchvision.models.squeezenet.SqueezeNet1_0_Weights.IMAGENET1K_V1)
    model = freeze_model(model)
    # UNUSUAL LAYER
    model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
    # INIT LAYER PARAMETERS
    nn.init.xavier_uniform_(model.classifier[1].weight)
    model.classifier[1].bias.data.fill_(0)
    return model

#-----------------------------------
class ImageFolderDataset(pl.LightningDataModule):
    def __init__(self, batch_size=32, workers=0, dataset_dir = dataset_directory, device = "cpu"):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch = batch_size
        self.num_workers = workers
        self.device = device
        self.g = torch.Generator(device = self.device )
        self.persistance = False
        if workers > 0:
            self.persistance = True

    def set_transforms(self,train_transform, test_transform):
        self.train_transform = train_transform
        self.test_transform = test_transform

    def train_dataloader(self):
        data = datasets.ImageFolder(self.dataset_dir / 'train',self.train_transform)
        self.names_classes = data.classes
        return torch.utils.data.DataLoader(data,
                                           batch_size=self.batch,
                                           shuffle=True,
                                           num_workers=self.num_workers,
                                           persistent_workers = self.persistance,
                                           generator = self.g)
    
    def test_dataloader(self):
        data = datasets.ImageFolder(self.dataset_dir  / 'test', self.test_transform)
        return torch.utils.data.DataLoader(data,
                                           batch_size=self.batch,
                                           shuffle=False,
                                           num_workers=self.num_workers,
                                           persistent_workers = self.persistance)
    
    def val_dataloader(self):
        data = datasets.ImageFolder(self.dataset_dir  / 'val',self.test_transform)
        return torch.utils.data.DataLoader(data,
                                           batch_size=self.batch,
                                           shuffle=False,
                                           num_workers=self.num_workers,
                                           persistent_workers = self.persistance)
        