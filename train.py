import os
import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer

dataset_path = '../../../disk2/icchiu/inf_dataset/video_v2'
save_name = "temporal_conv"
load_name = "base_cont"
is_temporal = True

model = Unet3D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    temporal_conv = is_temporal,
)

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    num_frames = 64,
    timesteps = 1000,   # number of steps
    loss_type = 'l2'    # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    dataset_path,                         # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
    seq_length = 64,
    height = 64,
    width = 64,
    train_batch_size = 4,
    train_lr = 1e-4,
    save_and_sample_every = 1000,      # 1000
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                        # turn on mixed precision
    results_folder = f"results/{save_name}",
    load_folder = f"results/{load_name}"
)

# trainer.load()

trainer.train()