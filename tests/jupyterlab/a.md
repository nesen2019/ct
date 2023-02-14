---
author: Bangguo Chen
title: a
description: aa
slug: a
date: 2022-01-15
categories:
    - a
tags: 
    - docker
    - docker123123
---

- [H1](#h1)
  - [H2](#h2)
    - [H3](#h3)
      - [H4](#h4)
  - [panda](#panda)
  - [add ${}file](#add-file)



```python
!pwd
```

    /root/Documents/ct/tests/jupyterlab



```python
ia
```

    The Zen of Python, by Tim Peters
    
    Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex.
    Complex is better than complicated.
    Flat is better than nested.
    Sparse is better than dense.
    Readability counts.
    Special cases aren't special enough to break the rules.
    Although practicality beats purity.
    Errors should never pass silently.
    Unless explicitly silenced.
    In the face of ambiguity, refuse the temptation to guess.
    There should be one-- and preferably only one --obvious way to do it.
    Although that way may not be obvious at first unless you're Dutch.
    Now is better than never.
    Although never is often better than *right* now.
    If the implementation is hard to explain, it's a bad idea.
    If the implementation is easy to explain, it may be a good idea.
    Namespaces are one honking great idea -- let's do more of those!


# H1

## H2

### H3

#### H4

- 123
- 234
- 345




```python

```


```python
import a
```


```python
a.printa()
```

    Print a...


## panda

![panda](https://alifei03.cfp.cn/creative/vcg/800/version23/VCG21gic15754169.jpg)


```python
aa
```

## add ${}file


```python
import glob 
import os 
import shutil
```


```python
ct_dir = "/root/Documents/ct"
_dirs = glob.glob(os.path.join(ct_dir, "**/*"), recursive=True)

fstr = """\
---
author: Bangguo Chen
title: {title}
description: {title}
slug: slug-{title}
date: 2022-01-15
categories:
tags: 
---
"""


for _dir in _dirs:
    if os.path.isdir(_dir):
        title = os.path.basename(_dir)
        print(_dir)
        file = os.path.join(_dir, f"{title}.md")
        with open(file, "w") as f:
            f.write(fstr.format(title=title))
            f.write("\n")


```

    /root/Documents/ct/Associates

