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


class Model4AAAI(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.text_encoder = BertEncoder()
        self.image_encoder = VITB16_encoder()

        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        self.fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 8)
        )

        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x):
        text_embedding = self.text_encoder(x['texts']['input_ids'], x['texts']['attention_mask'])
        image_embedding = self.image_encoder(x['imgs'])[:,0,:]
        fused_features = torch.stack([image_embedding, image_embedding], dim=0)
        attn_output, _ = self.attention(fused_features, fused_features, fused_features)
        fused_features = attn_output.mean(dim=0)  # 对两个模态的特征进行平均
        output = self.fc(fused_features)
        return output
    
    def training_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self.loss(pred, batch["labels"])
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