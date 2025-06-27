import re
import cv2
import pdb
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from glob import glob
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from PIL import Image
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence


class MedDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, label_paths, des_path, image_transform=None):
        super().__init__()
        self.img_paths = glob(img_path+"/images/*.jpg")
        self.labels = {} # key是图片名称，values是图片类别

        # 
        for label_path in label_paths:
            with open(label_path, 'r') as f:
                reader = f.readlines()
                for line in reader:
                    img_name, lable = line.replace("\n", "").split(',')
                    self.labels[img_name] = int(lable)
        self.des = {}
        with open(des_path, 'r') as f:
            desc_data = json.load(f)

        for k, v in desc_data.items():
            data_name = os.path.basename(k)
            self.des[data_name] = v

        self.image_transform = image_transform


    def __len__(self, ):
        return len(self.img_paths)

    def remove_chinese_and_punctuation(self, text):
        # 正则表达式：匹配所有中文字符和中文标点符号
        # \u4e00-\u9fa5：匹配常用汉字
        # \u3000-\u303F：匹配中文标点符号
        # \uff00-\uffef：匹配全角字符，包括一些标点
        return re.sub(r'[\u4e00-\u9fa5\u3000-\u303F\uff00-\uffef]', '', text)

    def __getitem__(self, index):
        subject_name = os.path.basename(self.img_paths[index])
        result = {
            'img': self.image_transform(Image.open(self.img_paths[index])),
            'label': self.labels[subject_name],
            'text_raw': self.des[subject_name]  # 返回原始文本，collate_fn中统一tokenize
        }
        return result



tokenizer = BertTokenizer.from_pretrained('/mnt/data2/zzixuantang/AAAI/pretrain_model/BERT_pretain')
def collate_fn(batch):
    # 分离不同字段
    imgs = [item['img'] for item in batch]
    labels = [item['label'] for item in batch]
    texts = [item['text_raw'] for item in batch]  # 假设返回的是原始文本

    imgs = torch.stack(imgs, dim=0)
    labels = torch.tensor(labels)

    # 处理文本：统一tokenize并填充
    text_encodings = tokenizer(
        texts,
        padding=True,          # 自动填充到批次内最长长度
        truncation=True,       # 截断超长文本
        max_length=512,       # 设置最大长度
        return_tensors='pt',   # 返回PyTorch张量
    )

    # 返回批处理结果
    return {
        'imgs': imgs,
        'labels': labels,
        'texts': text_encodings  # 包含input_ids, attention_mask等
    }

# 1. 定义 LightningDataModule
class DataModule(pl.LightningDataModule):
    def __init__(self, data_root, train_img_dir_name, test_img_dir_name, batch_size=64, num_workers=4):
        super().__init__()
        self.data_dir = data_root

        self.train_img_dir = os.path.join(data_root, train_img_dir_name)
        self.test_img_dir = os.path.join(data_root, test_img_dir_name)

        self.batch_size = batch_size
        self.num_workers = num_workers
        

        # 定义数据增强和转换
        self.transform = transforms.Compose([
            transforms.RandomRotation(10),  # 随机旋转
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
            transforms.ToTensor(),
        ])
        
        # 测试集不需要数据增强
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    

    def setup(self, stage=None):
        # 分配训练/验证/测试数据
        if stage == 'fit' or stage is None:
            image_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),  # 随机裁剪到224x224大小
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.ToTensor(),              # 转换为Tensor格式
            ])
            self.train_dataset = MedDataset(
                '/mnt/data2/zzixuantang/classfier_convNext/data/00-HAM10000/train', 
                glob('/mnt/data2/zzixuantang/classfier_convNext/data/00-HAM10000/label/*.csv'), 
                '/mnt/data2/zzixuantang/classfier_convNext/data/00-HAM10000/03-临床症状-trainset-4o-WithoutcolorAndSize.json',
                image_transform = image_transform,
            )
        
        if stage == 'test' or stage == 'val':
            image_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),  # 随机裁剪到224x224大小
                transforms.ToTensor(),              # 转换为Tensor格式
            ])
            self.test_dataset = MedDataset(
                '/mnt/data2/zzixuantang/classfier_convNext/data/00-HAM10000/test', 
                glob('/mnt/data2/zzixuantang/classfier_convNext/data/00-HAM10000/label/*.csv'), 
                '/mnt/data2/zzixuantang/classfier_convNext/data/00-HAM10000/03-临床症状-testset-4o-WithoutcolorAndSize.json',
                image_transform = image_transform,
            )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                         shuffle=True, num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn)

    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                         num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn)

    def validation_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                         num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn)
    

if __name__ == "__main__":
    pass