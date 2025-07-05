import pdb
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

import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# 3. 主程序
def main():

    # 加载配置文件
    config = load_config('config.yaml')
    
    exp_name = config['experiment']['experiment_name']
    # 初始化数据模块
    dm = DataModule(
        batch_size=config['train']['batch_size'], 
        data_root=config['data']['data_root'], 
        train_img_dir_name = 'train', 
        test_img_dir_name = 'test'
    )
    
    # 初始化模型
    model = Model4AAAI(learning_rate=config['train']['learning_rate'], config = config)
    
    # 定义回调函数
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=f"{config['experiment']['save_dir']}/{exp_name}/checkpoints/",
        filename='mnist-{epoch:02d}-{val_acc:.2f}',
        save_top_k=config['train']['save_topk'],
        mode='max'
    )
    
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # 定义日志记录器
    tb_logger = TensorBoardLogger(f'./exp/{exp_name}/logs/')
    
    # 初始化Trainer
    trainer = pl.Trainer(
        max_epochs=config['train']['epoch'],
        accelerator='auto',
        devices='auto',
        logger=[tb_logger],
        callbacks=[checkpoint_callback, lr_monitor],
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=config['train']['log_every_n_steps'],
        fast_dev_run=False,  # 设为True可以快速检查代码是否能运行
        overfit_batches=0,  # 设为>0可以用于调试过拟合
        check_val_every_n_epoch=config['train']['check_val_every_n_epoch']
    )
    # 训练模型
    trainer.fit(model, datamodule=dm)
    
    # # 测试模型
    # trainer.test(model, datamodule=dm)
    
    # 打印最佳模型路径
    print(f"最佳模型保存在: {checkpoint_callback.best_model_path}")

if __name__ == '__main__':
    main()