
import torch
from torch.utils.data import DataLoader
from hsi_dataset_25 import TestDataset
from architecture import model_generator
from model_0611 import Net
from utils import lightfield_resize_train, lightfield_angluarresize, cal_metrics
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2



def pad_to_multiple(x, multiple=11):
    *leading_dims, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    pad = (0, pad_w, 0, pad_h)
    x = x.view(-1, h, w)  # flatten leading dims
    x = F.pad(x, pad)
    new_h, new_w = x.shape[-2], x.shape[-1]
    x = x.view(*leading_dims, new_h, new_w)
    return x


def crop_CHWN_to_same_size(a, b):
    H = min(a.shape[1], b.shape[1])
    W = min(a.shape[2], b.shape[2])
    return center_crop_tensor(a, H, W), center_crop_tensor(b, H, W)

def center_crop_tensor(x, H, W):
    _, h, w, _ = x.shape
    start_h = (h - H) // 2
    start_w = (w - W) // 2
    return x[:, start_h:start_h+H, start_w:start_w+W, :]




os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')
torch.backends.cudnn.benchmark = True

# ======== setup ========
checkpoint_path = "check_point/edge/mprnet.pth"
data_root = '/data'  # 数据集根路径
batch_size = 1
save_root = '/home/code/MST/real_result/edge/mprnet'
os.makedirs(save_root, exist_ok=True)

# ======== load dataset ========
val_data = TestDataset(data_root=data_root, bgr2rgb=True)
val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
print(f"[✓] Num of testdata: {len(val_data)}")


model = model_generator('mprnet',checkpoint_path).to(device)
sr_net = Net(1, 2).to(device)

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])


sr_net.load_state_dict(checkpoint['sr_net_state_dict'])

model.eval()
sr_net.eval()


psnr_list = []
ssim_list = []

with torch.no_grad():
    for i, (input, target) in enumerate(val_loader):

        input = input.to(device).float()    # 1,3,121,H,W
        target = target.to(device).float()  # 1,28,25,H,W
        input = pad_to_multiple(input, 11)
        target = pad_to_multiple(target, 11)

        input_resize = lightfield_resize_train(input, 1)  # 1,3,121,16,16
        rgb = input_resize.cpu().numpy()
        rgb = rgb[0,:,5,:,:].transpose(1,2,0)
        plt.imshow(rgb)
        plt.title("low resolution input rgb image (view : 5,0)")
        plt.savefig("input_rgb.png", dpi=2000, bbox_inches='tight')
        plt.show()

        B,C,V,H,W = input_resize.size()
        out = model(input_resize)  # 1,28,121,256
        b, c, v, hw = out.shape
        h = H
        w = W
        out = out.reshape(b, c, v, h, w)
        result_all = []
        psnr_scene = []
        ssim_scene = []
        for spectral_id in range(c):
            sr_input = out[:, spectral_id]             # 1,121,16,16
            sr_label = target[:, spectral_id]          # 1,25,H,W

            result = sr_net(sr_input)                  # 1,121,176,176
            result_resize = lightfield_angluarresize(result, 5)  # 1,H,W,25
            sr_label = sr_label.permute(0, 2, 3, 1)             # 1,H,W,25

            result_resize = result_resize / result_resize.max()
            sr_label = sr_label / sr_label.max()
            result_resize = torch.clamp(result_resize, 0, 1)
            sr_label = torch.clamp(sr_label, 0, 1)
            result_all.append(result_resize)

            psnr, ssim = cal_metrics(sr_label, result_resize)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            psnr_scene.append(psnr)
            ssim_scene.append(ssim)
        result_all = torch.cat(result_all, dim=0).cpu().numpy()
        num_channels = result_all.shape[0]
        cols = 4
        rows = (num_channels + cols - 1) // cols  #

        plt.figure(figsize=(15, 3 * rows))
        for i in range(num_channels):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(result_all[i, :, :, 5], cmap='gray')
            plt.title(f'Channel {i}')
            plt.axis('off')


        plt.tight_layout()
        plt.savefig("channels.png", dpi=2000, bbox_inches='tight')
        plt.show()

        #High-resolution RGB image rendered from spectral data
        dataCube = result_all[:,:,:,5].astype(np.float32)
        r = dataCube[17, :, :]
        g = dataCube[8, :, :]
        b = dataCube[1, :, :]


        h, w = b.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        r_max = np.percentile(r, 98)
        g_max = np.percentile(g, 98)
        b_max = np.percentile(b, 98)

        rgb[:, :, 0] = np.clip((r / r_max * 255 * 0.77), 0, 255).astype(np.uint8)
        rgb[:, :, 1] = np.clip((g / g_max * 255 * 0.9), 0, 255).astype(np.uint8)
        rgb[:, :, 2] = np.clip((b / b_max * 255 * 0.84), 0, 255).astype(np.uint8)

        plt.imshow(rgb)
        plt.title("High-resolution RGB image rendered from spectral data (view: 5,0)")
        plt.show()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite('GT_rgb.png',bgr)

    data_array = np.array(psnr_list)
    averages = data_array.reshape(-1, 28).mean(axis=1)
    data_array1 = np.array(ssim_list)
    averages1 = data_array1.reshape(-1, 28).mean(axis=1)

    averages.tolist()
    averages1.tolist()
    print(f"\n PSNR: {sum(psnr_list)/len(psnr_list):.4f} dB")
    print(f"\n SSIM: {sum(ssim_list)/len(ssim_list):.4f}")
