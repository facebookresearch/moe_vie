# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import os
import random

import cv2
import decord
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class VideoRetrievalDataset(Dataset):
    def __init__(self, csv_path, dataset_dir, preprocessor, num_frames=8):
        self.data = pd.read_csv(csv_path)
        self.dataset_dir = dataset_dir


        self.preprocessor = preprocessor
        self.num_frames = num_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_id = self.data["video_id"].values[index]
        sentence = self.data["sentence"].values[index]
        video_path = os.path.join(self.dataset_dir, "{}.mp4".format(video_id))

        images = self._load_video(video_path)

        images = [
            (
                self.preprocessor(image.convert("RGB"))
                if image.mode == "L"
                else self.preprocessor(image)
            )
            for image in images
        ]

        return images, [sentence]

    def _load_video(self, media_path):
        vr = decord.VideoReader(media_path)
        total_frames = len(vr)
        frame_indices = [
            int(i * (total_frames - 1) / (self.num_frames - 1))
            for i in range(self.num_frames)
        ]

        try:
            images = vr.get_batch(frame_indices).asnumpy()
        except Exception as e:
            cap = cv2.VideoCapture(media_path)
            images = []
            for pos in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    images.append(rgb_frame)
                else:
                    break

        images = [Image.fromarray(image) for image in images]

        return images

