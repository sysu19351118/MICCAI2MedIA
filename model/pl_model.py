import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .BERT import BertEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR
from .VITB16 import VITB16_encoder
import pdb
from .block.mamba_block import MambaFusionTransformer


class BaseLine(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_encoder = BertEncoder()
        self.image_encoder = VITB16_encoder()
        self.fc = nn.Linear(512, 8)
        self.mamba_fusion = MambaFusionTransformer(
            num_layer=config['model']['baseline']['mamba_fusion']['num_layer'],
            indims=config['model']['baseline']['mamba_fusion']['indims'],
            outdims=config['model']['baseline']['mamba_fusion']['outdims'],
            indims_fusion=config['model']['baseline']['mamba_fusion']['indims_fusion'],
            outdims_fusion=config['model']['baseline']['mamba_fusion']['outdims_fusion'],
            dim_model = config['model']['baseline']['mamba_fusion']['dim_model'],
        )

    def forward(self, x):
        text_embedding = self.text_encoder(x['texts']['input_ids'], x['texts']['attention_mask'])
        image_embedding = self.image_encoder(x['imgs'])
        multimodal_embedding = self.mamba_fusion(image_embedding, text_embedding)
        output = self.fc(multimodal_embedding)
        return output

class Model4AAAI(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, config = None):
        super().__init__()
        self.save_hyperparameters()
        self.loss = nn.CrossEntropyLoss()
        self.net = BaseLine(config)
        self.config = config

    
    def forward(self, x):
        return self.net(x)

    
    def training_step(self, batch, batch_idx):
        # 前向传播
        pred = self(batch)
        
        # 计算损失
        loss = self.loss(pred, batch["labels"])
        
        # 计算准确率
        pred_labels = pred.argmax(dim=-1)  # 假设是分类任务，获取预测类别
        correct = (pred_labels == batch["labels"]).float().sum()
        total = batch["labels"].size(0)
        accuracy = correct / total
        
        # 记录指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self.loss(pred, batch["labels"])
        pred_labels = torch.argmax(pred, dim=1)
        acc = (pred_labels == batch["labels"]).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}  # 可省略

    def test_step(self, batch, batch_idx):
        pass
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        
        scheduler = {
            "scheduler": CosineAnnealingLR(
                optimizer,
                T_max=10,  # 最大迭代次数（通常设为总 epoch 数）
                eta_min=1e-6,  # 最小学习率
            ),
            "interval": "epoch",  # 按 epoch 更新
            "frequency": 1,       # 每个 epoch 更新一次
        }
        
        return [optimizer], [scheduler]