#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/5/15 15:34
# @Author  : zhangchao, Wenrui Han
# @File    : trainer.py
# @Software: VS Code
# @Email   : zhangchao5@genomics.cn, hwr9912@163.com
import os
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from spatialid import SpatialModel, KDLoss, DNNModel, MultiCEFocalLoss, DNNDataset


class Base:
    def __init__(self, device):
        self.model = None
        self.optimizer = None
        self.scaler = None
        self.scheduler = None
        self.criterion = None
        if device > -1 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device}")
        else:
            self.device = torch.device("cpu")

    def set_model(self, *args, **kwargs):
        pass

    def set_optimizer(self, *args, **kwargs):
        pass

    @staticmethod
    def load_checkpoint(path):
        assert os.path.exists(path)
        # 这一步需要加 weights_only=False
        # UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. 
        checkpoint = torch.load(path, weights_only=False)
        assert isinstance(checkpoint, dict)
        assert 'model' in checkpoint.keys()
        print(f"The checkpoint was saved: {checkpoint.keys()}")
        return checkpoint

    def save_checkpoint(self, path: str, state: dict = None):
        base_info = {'model': self.model}
        if state is not None:
            base_info.update(state)
        torch.save(base_info, path)


class SpatialTrainer(Base):
    def __init__(self, input_dim, num_classes, lr, weight_decay, device):
        super(SpatialTrainer, self).__init__(device=device)
        self.set_model(input_dim, num_classes)
        self.set_optimizer(lr=lr, weight_decay=weight_decay)

    def set_model(self, input_dim, num_classes):
        gae_dim, dae_dim, feat_dim = [32, 8], [100, 20], 64
        self.model = SpatialModel(input_dim, num_classes, gae_dim, dae_dim, feat_dim).to(self.device)
        self.criterion = KDLoss(1)

    def set_optimizer(self, lr=0.01, weight_decay=0.0001, **kwargs):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=.95)

    def train(self, data, epochs, w_cls, w_dae, w_gae):
        self.model.train()
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            data = data.to(self.device, non_blocking=True)
            inputs, targets = data.x, data.y
            edge_index = data.edge_index
            edge_weight = data.edge_weight

            outputs, dae_loss, gae_loss = self.model(inputs, edge_index, edge_weight)
            loss = w_cls * self.criterion(outputs, targets) + w_dae * dae_loss + w_gae * gae_loss
            train_loss = loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total = targets.size(0)
            predictions = outputs.argmax(1)
            correct = predictions.eq(targets.argmax(1)).sum().item()
            self.scheduler.step()
            process_time = time.time() - start_time
            accuracy = correct / total * 100.0
            print('\r  [Epoch %3d] Loss: %.5f, Time: %.2f s, Psuedo-Acc: %.2f%%'
                  % (epoch, train_loss, process_time, accuracy), flush=True, end="")

    @torch.no_grad()
    def infer(self, data):
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            inputs = data.x
            edge_index = data.edge_index
            edge_weight = data.edge_weight
            outputs, _, _ = self.model(inputs, edge_index, edge_weight)
            predictions = outputs.argmax(1)
        predictions = predictions.detach().cpu().numpy()
        return predictions


class DnnTrainer(Base):
    def __init__(self, input_dims, label_names, device=1, lr=3e-4, weight_decay=1e-6, gamma=2, alpha=.25, reduction="mean", save_path="./dnn.bgi"):
        super(DnnTrainer, self).__init__(device=device)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.input_dims = input_dims
        self.n_types = len(label_names)
        self.label_names = label_names
        self.save_path = save_path
        self.set_model(input_dims, hidden_dims=1024, output_dims=self.n_types, gamma=gamma, alpha=alpha, reduction=reduction)
        self.set_optimizer(lr=lr, weight_decay=weight_decay)

    def set_model(self, input_dims, hidden_dims, output_dims, gamma, alpha, reduction, **kwargs):
        self.model = DNNModel(input_dims, hidden_dims, output_dims).to(self.device)
        self.criterion = MultiCEFocalLoss(class_num=self.n_types, gamma=gamma, alpha=alpha, reduction=reduction)

    # def set_optimizer(self, lr=3e-4, weight_decay=1e-6, **kwargs):
    #     self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    def set_optimizer(self, lr=3e-4, weight_decay=1e-6, milestones=(50, 100, 150), lr_gamma=0.1, **kwargs):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # 设置MultiStepLR
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=list(milestones), gamma=lr_gamma)
        # 训练循环中需要

    def train(self, data, ann_key, marker_genes=None, batch_size=4096, epochs=200):
        # 内存爆炸
        dataset = DNNDataset(adata=data, ann_key=ann_key, marker_genes=marker_genes)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=16)
        # 存储训练中模型功效评估指标
        history = []

        self.model.train()
        best_loss = np.inf
        for epoch in tqdm(range(epochs)):
            epoch_acc = []
            epoch_loss = []
            # 参数迭代
            for idx, data in enumerate(loader):
                # 下面是内存开销大头
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.long().to(self.device)
                output = self.model(inputs)
                loss = self.criterion(output, targets)
                train_loss = loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total = targets.size(0)
                prediction = output.argmax(1)
                correct = prediction.eq(targets).sum().item()

                accuracy = correct / total * 100.
                epoch_acc.append(accuracy)
                epoch_loss.append(train_loss)
            # MultiStepLR 更新
            if self.scheduler is not None:
                self.scheduler.step()
            # 实时输出
            now_time = time.strftime('%Y-%m-%d %H:%M:%S')
            mean_loss = np.mean(epoch_loss)
            mean_acc = np.mean(epoch_acc)
            tqdm.write(
                f"  [{now_time}] "
                f"Epoch: {epoch + 1:3d} "
                f"Loss: {mean_loss:.5f}, "
                f"acc: {mean_acc:.2f}%"
            )
            # 保存参数
            if np.mean(epoch_loss) < best_loss:
                best_loss = np.mean(epoch_loss)
                state = {
                    'marker_genes': marker_genes,
                    'batch_size': batch_size,
                    'label_names': self.label_names
                }
                self.save_checkpoint(self.save_path, state)
            # 保存训练记录
            history.append({
                "epoch": epoch + 1,
                "time": now_time,
                "loss": mean_loss,
                "acc": mean_acc,
                "best_loss": best_loss,
            })

        print("\n validation model: ")
        checkpoint = self.load_checkpoint(self.save_path)
        with torch.no_grad():
            best_model = checkpoint["model"]
            best_model.to(self.device)
            best_model.eval()
            val_acc = []
            for idx, data in enumerate(loader):
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.long().to(self.device)
                outputs = best_model(inputs)

                total = targets.size(0)
                prediction = outputs.argmax(1)
                correct = prediction.eq(targets).sum().item()
                accuracy = correct / total * 100.
                val_acc.append(accuracy)
            print(f"\n  [{time.strftime('%Y-%m-%d %H:%M:%S')} total accuracy: {np.mean(val_acc):.2f}%]")
        
        # 训练结束后保存 csv
        history_df = pd.DataFrame(history)
        # 保存到 self.save_path 的同级目录下
        history_df.to_csv(os.path.join(os.path.dirname(self.save_path), "train_history.csv"), index=False)
