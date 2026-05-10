#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/28 10:46
# @Author  : zhangchao
# @File    : loader.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import numpy as np
import scipy.sparse as sp

from torch.utils.data import Dataset


# class DNNDataset(Dataset):
#     def __init__(self, adata, ann_key, marker_genes=None):
#         self.adata = adata
#         self.shape = adata.shape
#         self.ann_key = ann_key
#         if sp.issparse(adata.X):
#             adata.X = adata.X.toarray()

#         if marker_genes is None:
#             data = adata.X
#         else:
#             gene_indices = adata.var_names.get_indexer(marker_genes)
#             data = np.pad(adata.X, ((0, 0), (0, 1)))[:, gene_indices].copy()

#         norm_factor = np.linalg.norm(data, axis=1, keepdims=True)
#         norm_factor[norm_factor == 0] = 1
#         self.data = data / norm_factor

#     def __len__(self):
#         return self.shape[0]

#     def __getitem__(self, idx):
#         x = self.data[idx].squeeze()
#         y = self.adata.obs[self.ann_key].cat.codes[idx]
#         return x, y

class DNNDataset(Dataset):
    def __init__(self, adata, ann_key, marker_genes=None):
        self.adata = adata
        self.shape = adata.shape
        self.ann_key = ann_key

        if marker_genes is None:
            x = adata.X
        else:
            gene_indices = adata.var_names.get_indexer(marker_genes)
            gene_indices = gene_indices[gene_indices >= 0]
            x = adata[:, gene_indices].X

        if sp.issparse(x):
            x = x.toarray()

        x = x.astype(np.float32, copy=False)

        norm_factor = np.linalg.norm(x, axis=1, keepdims=True)
        norm_factor[norm_factor == 0] = 1

        self.data = x / norm_factor
        self.labels = adata.obs[ann_key].cat.codes.to_numpy()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y