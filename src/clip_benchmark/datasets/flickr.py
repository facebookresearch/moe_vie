# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import glob
import os
from collections import defaultdict
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image
from torchvision.datasets import VisionDataset

class Flickr(VisionDataset):

    def __init__(
        self,
        root: str,
        ann_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)
        data = defaultdict(list)
        with open(ann_file) as fd:
            fd.readline()
            for line in fd:
                line = line.strip()
                if line:
                    img, caption = line.strip().split(".jpg,")
                    img = img + ".jpg"
                    data[img].append(caption)
        self.data = list(data.items())
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, captions = self.data[index]

        img = Image.open(os.path.join(self.root, img)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        target =  captions
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self) -> int:
        return len(self.data)
