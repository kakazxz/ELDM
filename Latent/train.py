import os
import torch
from L2L import Unet, GaussianDiffusionImg, Trainer,ConditionalEncoder,DenoiserNet
from AE import PerceptualAutoencoder
from loader import ct_latent_dataset
if __name__ == "__main__":
    ck_path = "/media/Data/aapm/data/aev5_dim64/best_psnr.pth"
    data_set = "/media/Data/aapm/npy_img"
    data_set_latent = "/media/Data/aapm/ae_dim64/"
    result_folder="/media/Data/aapm/aapm_latent/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_AE = PerceptualAutoencoder(latent_dim=64,dim=32)
    model_AE.load_state_dict(torch.load(ck_path))
    img_decoder=model_AE.image_decoder

    model = DenoiserNet(in_channels=64, cond_channels=64, out_channels=64, dim=64, latent_dim=128,).cuda()

    c_encoder = ConditionalEncoder(dim=32).cuda()

    train_set= ct_latent_dataset(mode='train', load_mode=0,
                                 saved_path_1=data_set,
                                 saved_path_2=data_set, test_patient='L506',
               patch_n=False, patch_size=False, transform=None)

    val_set= ct_latent_dataset(mode='test', load_mode=0,
                               saved_path_1=data_set,
                                 saved_path_2=data_set_latent, test_patient='L506',
               patch_n=False, patch_size=False, transform=None)

    diffusion = GaussianDiffusionImg(
        model,
        c_encoder,
        img_decoder,
        image_size = 64,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 100,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = 'l1' ,           # L1 or L2
        objective="pred_x0"
    ).cuda()

    trainer = Trainer(
        diffusion,
        r'',
        r'',
        train_batch_size = 128,
        train_lr =1e-3,# 8e-5,
        train_num_steps = 300000,         # total training steps
        save_and_sample_every=1000,
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = False,                      # turn off mixed precision
        num_workers=4,
        num_samples=4,
        results_folder=result_folder,
        train_set=train_set,
        val_set=val_set
    )

    trainer.train()
