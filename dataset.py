import json
import os

import einops
from pathlib import Path, PurePosixPath
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from color import *
from utils import video_tensor_to_mp4

class VideoDataset_INF0(Dataset):

    def __init__(
        self,
        dataset_dir = "",
        seq_length = 64,
        height = 64,
        width = 64,
        min_spacing = 1,
        max_spacing = 1,
        x_flip = False,
        mode = "train"
    ):
        self.dataset_dir = dataset_dir
        self.seq_length = seq_length
        self.height = height
        self.width = width
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing
        self.x_flip = x_flip
        self.mode = mode

        assert self.seq_length >= 1

        if self.mode == "train":
            self.dataset_dir = self.dataset_dir+"/train"
        elif self.mode == "val":
            self.dataset_dir = self.dataset_dir+"/validation"

        self.dataset_path = Path(self.dataset_dir)
        assert self.dataset_path.is_dir(), self.dataset_path

        self.min_video_length = self.seq_length

        self.video_paths=[]

        for dir in os.listdir(self.dataset_dir):
            dir_name = self.dataset_dir+"/"+dir
            frame_names = [] 
            for img in sorted(os.listdir(dir_name)):
                frame_names.append(img)

            frame_names = frame_names[:-4]
            
            if len(frame_names) >= self.min_video_length:
                video_tuple = (dir_name,frame_names)
                self.video_paths.append(video_tuple)
        
        print_g(f"availabe {self.mode} data:{len(self.video_paths)}") 

        self._zipfiles = {}

    def sample_frame_names(self, frame_names):
        max_spacing = (
            1 if self.seq_length == 1 else min(self.max_spacing, (len(frame_names) - 1) // (self.seq_length - 1))
        )
        spacing = torch.randint(self.min_spacing, max_spacing + 1, size=()).item()

        frame_span = (self.seq_length - 1) * spacing + 1
        max_start_index = len(frame_names) - frame_span
        start_index = torch.randint(max_start_index + 1, size=()).item()

        frame_names = frame_names[start_index : start_index + frame_span : spacing]
        return frame_names, spacing
    
    def center_crop_and_resize(self,frame, height, width):

        # Measures by what factor height and width are larger/smaller than desired.
        height_scale = frame.height / height
        width_scale = frame.width / width

        # Center crops whichever dimension has a greater scale factor.
        if height_scale > width_scale:
            crop_height = height * width_scale
            y0 = (frame.height - crop_height) // 2
            y1 = y0 + crop_height
            frame = frame.crop((0, y0, frame.width, y1))

        elif width_scale >= height_scale:
            crop_width = width * height_scale
            x0 = (frame.width - crop_width) // 2
            x1 = x0 + crop_width
            frame = frame.crop((x0, 0, x1, frame.height))

        # Resizes to desired height and width.
        frame = frame.resize((width, height), Image.LANCZOS)
        return frame

    def read_frame(self, frame_path):

        frame = Image.open(frame_path)
        frame = self.center_crop_and_resize(frame,self.height,self.width)
        # frame.save(f"test_folder/test.png")
        frame = np.array(frame)
        frame = torch.from_numpy(frame)
        frame = einops.rearrange(frame, "h w c -> c h w")
        # frame = 2 * frame.to(torch.float32) / 255 - 1
        frame = frame.to(torch.float32) / 255
        return frame

    def __getitem__(self, index):
        clip_path, frame_names = self.video_paths[index]
        frame_names, spacing = self.sample_frame_names(frame_names)
        frame_paths = [str(PurePosixPath(clip_path).joinpath(frame_name)) for frame_name in frame_names]
        frames = [self.read_frame(frame_path) for frame_path in frame_paths]
        video = torch.stack(frames, dim=1)

        se3 = np.load(f"{clip_path}/se3.npy")

        if self.x_flip and torch.rand(()).item() < 0.5:
            video = video.flip(dims=(-1,))

        return video

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getstate__(self):
        return dict(self.__dict__, _zipfiles={})
    

class VideoDataset_ACID(Dataset):

    def __init__(
        self,
        dataset_dir = "",
        seq_length = 64,
        height = 64,
        width = 64,
        min_spacing = 1,
        max_spacing = 1,
        x_flip = False,
        mode = "train"
    ):
        self.dataset_dir = dataset_dir
        self.seq_length = seq_length
        self.height = height
        self.width = width
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing
        self.x_flip = x_flip
        self.mode = mode

        assert self.seq_length >= 1

        if self.mode == "train":
            self.dataset_dir = self.dataset_dir+"/train"
        elif self.mode == "val":
            self.dataset_dir = self.dataset_dir+"/validation"

        self.dataset_path = Path(self.dataset_dir)
        assert self.dataset_path.is_dir(), self.dataset_path

        self.min_video_length = self.seq_length

        self.video_paths=[]

        for dir in os.listdir(self.dataset_dir):
            dir_name = self.dataset_dir+"/"+dir
            frame_names = [] 
            for img in sorted(os.listdir(dir_name)):
                frame_names.append(img)

            frame_names = frame_names[:-1]
            
            if len(frame_names) >= self.min_video_length:
                video_tuple = (dir_name,frame_names)
                self.video_paths.append(video_tuple)
        
        print_g(f"availabe {self.mode} data:{len(self.video_paths)}") 

        self._zipfiles = {}

    def sample_frame_names(self, frame_names):
        max_spacing = (
            1 if self.seq_length == 1 else min(self.max_spacing, (len(frame_names) - 1) // (self.seq_length - 1))
        )
        spacing = torch.randint(self.min_spacing, max_spacing + 1, size=()).item()

        frame_span = (self.seq_length - 1) * spacing + 1
        max_start_index = len(frame_names) - frame_span
        start_index = torch.randint(max_start_index + 1, size=()).item()

        frame_names = frame_names[start_index : start_index + frame_span : spacing]
        return frame_names, spacing
    
    def center_crop_and_resize(self,frame, height, width):
        #* 去除inf nature 黑邊
        frame = frame.crop((0,(1/6)*frame.height,frame.width,(5/6)*frame.height))

        # Measures by what factor height and width are larger/smaller than desired.
        height_scale = frame.height / height
        width_scale = frame.width / width

        # Center crops whichever dimension has a greater scale factor.
        if height_scale > width_scale:
            crop_height = height * width_scale
            y0 = (frame.height - crop_height) // 2
            y1 = y0 + crop_height
            frame = frame.crop((0, y0, frame.width, y1))

        elif width_scale >= height_scale:
            crop_width = width * height_scale
            x0 = (frame.width - crop_width) // 2
            x1 = x0 + crop_width
            frame = frame.crop((x0, 0, x1, frame.height))

        # Resizes to desired height and width.
        frame = frame.resize((width, height), Image.LANCZOS)
        return frame

    def read_frame(self, frame_path):

        frame = Image.open(frame_path)
        frame = self.center_crop_and_resize(frame,self.height,self.width)
        # frame.save(f"test_folder/test.png")
        frame = np.array(frame)
        frame = torch.from_numpy(frame)
        frame = einops.rearrange(frame, "h w c -> c h w")
        # frame = 2 * frame.to(torch.float32) / 255 - 1
        frame = frame.to(torch.float32) / 255
        return frame

    def __getitem__(self, index):
        clip_path, frame_names = self.video_paths[index]
        frame_names, spacing = self.sample_frame_names(frame_names)
        frame_paths = [str(PurePosixPath(clip_path).joinpath(frame_name)) for frame_name in frame_names]
        frames = [self.read_frame(frame_path) for frame_path in frame_paths]
        video = torch.stack(frames, dim=1)

        se3 = np.load(f"{clip_path}/se3.npy")

        if self.x_flip and torch.rand(()).item() < 0.5:
            video = video.flip(dims=(-1,))

        return video,se3

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getstate__(self):
        return dict(self.__dict__, _zipfiles={})


# inf_dir = '../../../disk2/icchiu/inf_dataset/video_v2'

# inf_ds = VideoDataset_INF0(
#                 inf_dir,
#                 64,
#                 64, 
#                 64,
#                 x_flip = False,
#                 mode = "val"
#                 )

# video = inf_ds.__getitem__(5)
# video_tensor_to_mp4(video,"results/test/inf",0)
# print(se3)
# print()

# acid_dir = '../../../disk2/icchiu/acid_dataset/video'

# acid_ds = VideoDataset_ACID(
#                 acid_dir,
#                 64,
#                 64, 
#                 64,
#                 x_flip = False,
#                 mode = "train"
#                 )

# video,se3 = acid_ds.__getitem__(5)
# video_tensor_to_mp4(video,"results/test/acid",0)
# print(se3)
# print()

