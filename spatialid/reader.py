#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/5/15 15:34
# @Author  : zhangchao, Wenrui Han
# @File    : reader.py
# @Software: VS Code
# @Email   : zhangchao5@genomics.cn, hwr9912@163.com
import os
import scanpy as sc
from anndata import AnnData


def reader(data):
    if isinstance(data, AnnData):
        return data
    elif isinstance(data, str) and data.endswith("h5ad"):
        assert os.path.exists(data), ValueError(f"There was no data path: `{data}`!")
        return sc.read_h5ad(data)
    else:
        raise ValueError(f"Got an invalid data format, only support `str` and `AnnData`!")
