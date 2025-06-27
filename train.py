import sys
sys.path.append('/mnt/sda1/algorithom_code_summary/ToolsLearning/02-pytorch_lighting训练框架')
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

from dataset.pl_datset import DataModule
# from LightningModule.lightning_module import MNISTClassifier
from model.pl_model import Model4AAAI

# 3. 主程序
def main():
    # 初始化数据模块
    dm = DataModule(
        batch_size=24, 
        data_root='/mnt/data2/zzixuantang/classfier_convNext/data/00-HAM10000', 
        train_img_dir_name = 'train', 
        test_img_dir_name = 'test'
    )
    
    # 初始化模型
    model = Model4AAAI(learning_rate=1e-3)
    
    # 定义回调函数
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints/',
        filename='mnist-{epoch:02d}-{val_acc:.2f}',
        save_top_k=3,
        mode='max'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # 定义日志记录器
    tb_logger = TensorBoardLogger('logs/', name='mnist')
    csv_logger = CSVLogger('logs/', name='mnist')
    
    # 初始化Trainer
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='auto',
        devices='auto',
        logger=[tb_logger, csv_logger],
        callbacks=[checkpoint_callback, lr_monitor],
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=10,
        fast_dev_run=False,  # 设为True可以快速检查代码是否能运行
        overfit_batches=0,  # 设为>0可以用于调试过拟合
    )
    
    # 训练模型
    trainer.fit(model, datamodule=dm)
    
    # 测试模型
    trainer.test(model, datamodule=dm)
    
    # 打印最佳模型路径
    print(f"最佳模型保存在: {checkpoint_callback.best_model_path}")

if __name__ == '__main__':
    main()