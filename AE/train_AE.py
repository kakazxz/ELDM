import os
import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from AE import PerceptualAutoencoder
import torchvision.models as models


def tensor2np(tensor):
    return (tensor.detach().cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype(np.uint8)


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
epochs = 1000
save_ckt_path = "/data2/data/ae_dim64"
data_path = "/data2/data/npy_img"

os.makedirs(save_ckt_path, exist_ok=True)


model = PerceptualAutoencoder(in_channels=1, latent_dim=32, dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 损失函数
mse_loss = nn.MSELoss()

vgg19 = models.vgg19(pretrained=True).features.eval().to(device)

selected_layers = [1, 6, 11, 20]
layer_names = [str(i) for i in selected_layers]

vgg_model = nn.Sequential()
for name, layer in vgg19.named_children():
    vgg_model.add_module(name, layer)
    if name in layer_names:
        break
for param in vgg_model.parameters():
    param.requires_grad = False  #


def perceptual_loss(y_pred, y_true, vgg_model):

    if y_pred.shape[1] == 1:
        y_pred_vgg = y_pred.repeat(1, 3, 1, 1)
        y_true_vgg = y_true.repeat(1, 3, 1, 1)
    else:
        y_pred_vgg = y_pred
        y_true_vgg = y_true

    features_pred = vgg_model(y_pred_vgg)
    features_true = vgg_model(y_true_vgg)
    loss = nn.MSELoss()(features_pred, features_true)
    return loss



train_data_loader = get_loader_3(
    mode='train', load_mode=1, saved_path=data_path, test_patient='L506',
    patch_n=False, patch_size=False, transform=None,
    batch_size=batch_size, num_workers=6
)

val_data_loader = get_loader_3(
    mode='test', load_mode=1, saved_path=data_path, test_patient='L506',
    patch_n=False, patch_size=False, transform=None,
    batch_size=batch_size, num_workers=6
)

# 训练循环
best_psnr = 0.0
best_ssim = 0.0

for epoch in range(epochs):
    # ========== 训练部分 ==========
    model.train()
    total_list, mse_list, P_list = [], [], []
    for batch_idx, ndct_img in enumerate(train_data_loader):

        ndct_img = ndct_img.float().to(device)

        recon_image, z = model(ndct_img)
        loss_mse = mse_loss(recon_image, ndct_img)
        loss_perceptual = perceptual_loss(recon_image, ndct_img, vgg_model)

        lambda_vgg = 0.1
        total_loss = loss_mse + lambda_vgg * loss_perceptual

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_list.append(total_loss.item())
        mse_list.append(loss_mse.item())
        P_list.append(loss_perceptual.item())

    print(
        f"Epoch {epoch + 1}, All_Loss: {np.mean(total_list):.4f}, MSE_Loss: {np.mean(mse_list):.4f}, Perceptual_Loss: {np.mean(P_list):.4f}")

    # ========== 验证部分 ==========
    model.eval()
    psnr_list, ssim_list = []

    with torch.no_grad():
        for batch_idx, ndct_img in enumerate(val_data_loader):  # 修改：假设 loader 返回 (ndct_img,)
            ndct_img = ndct_img.float().to(device)

            recon_image, _ = model(ndct_img)  # 修改：获取重建图像

            # 转成 numpy 再算指标
            gt_np = tensor2np(ndct_img)
            pred_np = tensor2np(recon_image)

            for i in range(gt_np.shape[0]):  # 逐张图计算

                psnr = compare_psnr(gt_np[i], pred_np[i], data_range=255)
                ssim = compare_ssim(gt_np[i], pred_np[i], data_range=255, channel_axis=-1)
                psnr_list.append(psnr)
                ssim_list.append(ssim)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    print(f"[Val] Epoch {epoch + 1:03d}  PSNR: {avg_psnr:.4f}  SSIM: {avg_ssim:.4f}")

    # ========== 保存模型 ==========
    torch.save(model.state_dict(), os.path.join(save_ckt_path, "last.pth"))
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.state_dict(), os.path.join(save_ckt_path, "best_psnr.pth"))
    if avg_ssim > best_ssim:
        best_ssim = avg_ssim
        torch.save(model.state_dict(), os.path.join(save_ckt_path, "best_ssim.pth"))
