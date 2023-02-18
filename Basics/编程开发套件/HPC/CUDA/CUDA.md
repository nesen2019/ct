---
author: Bangguo Chen
title: CUDA
description: CUDA
slug: slug-CUDA
date: 2022-01-15
categories:
tags: 
---


# CUDA  

## nvcc用法
```sh
# nvcc
nvcc *.cu \
    --generate-code arch=compute_50,code=sm_50 \ # 
    -o hello-world


```

## cuda-base

### e01-HelloWorld from GPU device
















