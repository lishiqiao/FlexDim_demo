from __future__ import division
import hdf5storage
import torch
import torch.nn as nn
import logging
import numpy as np
import os
import scipy.ndimage
import math
from scipy.ndimage import gaussian_filter
import cv2
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim
from skimage import metrics
# from generate_rgb import rgb_image

device = torch.device('cuda')
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger
import torch
import os

def save_checkpoint(model_path, epoch, iteration, model, optimizer, psnr_value, ssim_value, sr_net=None):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'psnr': psnr_value,
        'ssim': ssim_value
    }

    # 可选：保存第二阶段网络 sr_net
    if sr_net is not None:
        state['sr_net_state_dict'] = sr_net.state_dict()

    filename = f'net_{epoch}epoch_psnr_{psnr_value:.4f}ssim_{ssim_value:.4f}pth'
    torch.save(state, os.path.join(model_path, filename))
    print(f"[✓] 模型已保存到：{os.path.join(model_path, filename)}")


# def save_checkpoint(model_path, epoch, iteration, model, optimizer,psnr_value,ssim_value):
#     state = {
#         'epoch': epoch,
#         'iter': iteration,
#         'state_dict': model.state_dict(),
#         'optimizer': optimizer.state_dict(),
#         'psnr':psnr_value,
#         'ssim':ssim_value
#     }
#
#     torch.save(state, os.path.join(model_path, 'net_%depoch_psnr_%.4fssim_%.4fpth' % (epoch, psnr_value, ssim_value)))

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        # label[label == 0] = 1e-6
        error = torch.abs(outputs - label) / label
        # mrae = torch.mean(error.view(-1))
        mrae = torch.mean(error.reshape(-1))

        mask = label != 0

        # 使用掩码筛选outputs和label
        outputs_masked = outputs[mask]
        label_masked = label[mask]

        # 计算相对误差
        error = torch.abs(outputs_masked - label_masked) / label_masked
        mrae = torch.mean(error)
        return mrae

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        # error = torch.from_numpy(error)
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.reshape(-1)))
        return rmse

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=1):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        mse = nn.MSELoss(reduce=False)
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()
    loss_csv.close

def normalize_image(image):
    # Normalize the image to the range [0, 1]
    # min_val = torch.min(image)
    max_val = torch.max(image)
    # norm_image = (image - min_val) / (max_val - min_val)
    norm_image = image / max_val
    return norm_image
def psnr_cal(file1, file2):
    # Load the .mat files


    # Assume the variable names in .mat files are 'pred' and 'spectral'
    image1 = file1
    image2 = file2

    # Check if the dimensions of the input images are the same
    # 检查输入图像的尺寸是否相同
    if image1.shape != image2.shape:
        raise ValueError("输入图像的尺寸必须相同")

    # 获取图像的维度
    batch, bands, rows, cols = image1.shape

    # 创建一个数组来存储每个波段的PSNR值
    psnr_values = np.zeros(bands * batch)

    # 循环遍历每个批次和每个波段
    index = 0
    for j in range(batch):
        for i in range(bands):
            # 提取当前波段的图像
            band_image1 = image1[j, i, :, :]
            band_image2 = image2[j, i, :, :]
            # band_image1 = normalize_image(band_image1)
            # band_image2 = normalize_image(band_image2)

            # 计算当前波段的PSNR
            mse = torch.mean((band_image1 - band_image2) ** 2)
            if mse == 0:
                psnr_values[index] = 100
            else:
                psnr_values[index] = 20 * torch.log10(1 / torch.sqrt(mse))

            index += 1

    # print('每个波段的PSNR值:')
    # print(psnr_values)

    # 计算并显示平均PSNR
    average_psnr = np.mean(psnr_values)
    # print('平均PSNR:')
    # print(average_psnr)
    return average_psnr
def pad_image(image, crop_size):
    _,_, h, w = image.shape
    pad_h = (crop_size - h % crop_size) % crop_size
    pad_w = (crop_size - w % crop_size) % crop_size
    padded_image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
    return padded_image, pad_h, pad_w

def crop_image(image, crop_size):
    _,_, h, w = image.shape
    crops = []
    for i in range(0, h, crop_size):
        for j in range(0, w, crop_size):
            crop = image[:, :, i:i + crop_size, j:j + crop_size]
            crops.append(crop)
    return crops
def stitch_image(crops, original_shape, crop_size, pad_h, pad_w):
    _, c, h, w = original_shape
    padded_h = h + pad_h
    padded_w = w + pad_w
    stitched_image = torch.zeros((1, c, padded_h, padded_w) ,device=device)
    idx = 0
    for i in range(0, padded_h, crop_size):
        for j in range(0, padded_w, crop_size):
            stitched_image[:, :, i:i + crop_size, j:j + crop_size] = crops[idx]
            idx += 1
    # 去掉填充部分，恢复到原始尺寸
    stitched_image = stitched_image[:, :, :h, :w]
    return stitched_image


def crop_image1(image, crop_size, stride):
    _, _, h, w = image.shape
    crops = []
    centers = []

    half_crop_size = crop_size // 2

    for i in range(half_crop_size, h - half_crop_size, stride):
        for j in range(half_crop_size, w - half_crop_size, stride):
            crop = image[:, :, i - half_crop_size:i + half_crop_size, j - half_crop_size:j + half_crop_size]
            crops.append(crop)
            centers.append((i, j))

    return crops, centers


def stitch_image1(crops, centers, original_shape, crop_size, center_size, pad_h, pad_w):
    _, c, h, w = original_shape
    padded_h = h + pad_h
    padded_w = w + pad_w
    stitched_image = torch.zeros((1, c, padded_h, padded_w), device=crops[0].device)

    half_crop_size = crop_size // 2
    half_center_size = center_size // 2

    for crop, (i, j) in zip(crops, centers):
        crop_center = crop[:, :, half_crop_size - half_center_size:half_crop_size + half_center_size,
                      half_crop_size - half_center_size:half_crop_size + half_center_size]
        stitched_image[:, :, i - half_center_size:i + half_center_size,
        j - half_center_size:j + half_center_size] = crop_center

    # Crop the padded area out
    stitched_image = stitched_image[:, :, :h, :w]

    return stitched_image






# def crop_image1(image, crop_size, stride):
#     """
#     将图像裁剪成多个重叠的块。
#
#     :param image: 输入图像 Tensor，形状为 (1, C, H, W)。
#     :param crop_size: 每个裁剪块的大小。
#     :param stride: 裁剪块之间的步幅，即重叠区域的大小。
#     :return: 裁剪块的列表。
#     """
#     _, _, h, w = image.shape
#     crops = []
#     for i in range(0, h - crop_size + 1, stride):
#         for j in range(0, w - crop_size + 1, stride):
#             crop = image[:, :, i:i + crop_size, j:j + crop_size]
#             crops.append(crop)
#     return crops ,i,j
def LF2multiview(psf2D, u, v):
    c, x_u, y_v = psf2D.shape  # 获取通道数
    x = x_u // u
    y = y_v // v

    psf_z = np.zeros((c, x, y, u, v))

    for uu in range(u):
        for vv in range(v):
            psf_z[:, :, :, uu, vv] = psf2D[:, uu::u, vv::v]

    psf_z = psf_z.reshape(c, x, y, u * v)
    psf_z = psf_z.transpose(3, 0, 1, 2)

    return psf_z
def ssim(img1, img2, window, window_size, val_range):
    window = window.to(img1.device)
    mu1 = F.conv2d(img1.unsqueeze(0).unsqueeze(0), window, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(img2.unsqueeze(0).unsqueeze(0), window, padding=window_size // 2, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1.unsqueeze(0).unsqueeze(0).pow(2), window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2.unsqueeze(0).unsqueeze(0).pow(2), window, padding=window_size // 2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1.unsqueeze(0).unsqueeze(0) * img2.unsqueeze(0).unsqueeze(0), window, padding=window_size // 2, groups=1) - mu1_mu2

    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def lightfield_views_torch(stitched_data):
    '''
            确保输入的是batch,channel,H,W 2,36,160,160
    '''
    # 确保数据在GPU上
    stitched_data = stitched_data.to('cuda')

    bh,C,x, y,  = stitched_data.shape
    u = 9
    v = 9

    # 初始化 light2D 数组 2 36 5 5 32 32
    light2D = torch.zeros((bh, C, u, v, x // u, y // v), dtype=stitched_data.dtype, device=stitched_data.device)
    # light2D = torch.zeros((bh,C, u, v, x // u, y // v), dtype=stitched_data.dtype, device=stitched_data.device)

    # 使用高级索引在GPU上进行快速处理
    # stitched_data1=stitched_data[:,:,0:315,0:315] #9_9
    stitched_data1 = stitched_data[:, :, 0:319, 0:319]  # 9_9
    for uu in range(u):
        for vv in range(v):
            light2D[:, :, uu, vv, :, :] = stitched_data1[:,:,uu::u, vv::v]

    return light2D
def lightfield_views_numpy(stitched_data):
    '''
    将数据重排列为 (batch, channels, u, v, rows, cols)
    '''
    bc,channels, x, y = stitched_data.shape
    u, v = 9, 9

    # Initialize the light field array
    light2D = np.zeros(( bc,channels, u, v, x // u, y // v), dtype=np.float32)

    # Reshape the image data to light field views
    for uu in range(u):
        for vv in range(v):
            light2D[ :,:, uu, vv, :, :] = stitched_data[ :,:, uu::u, vv::v]

    return light2D
def ssim_cal_test(file1, file2, window_size=11, val_range=1.0):
    """
    Calculate SSIM for each band and batch, then return the average SSIM.

    :param file1: The first input array (e.g., predicted image) of shape (batch, bands, rows, cols).
    :param file2: The second input array (e.g., ground truth image) of the same shape as file1.
    :param window_size: The size of the Gaussian window used for SSIM calculation.
    :param val_range: The range of the image values (default is 1.0 for normalized images).
    :return: The average SSIM value across all batches and bands.
    """

    # Check if the dimensions of the input images are the same
    if file1.shape != file2.shape:
        raise ValueError("输入图像的尺寸必须相同")
    file1 = file1.cpu()
    file2 = file2.cpu()
    # Get image dimensions
    file1 = lightfield_views_numpy(file1)
    file2 = lightfield_views_numpy(file2)

    bc,bands, u, v, rows, cols = file1.shape

    # Create an array to store SSIM values for each band
    ssim_values = np.zeros(bands *bc * u * v)

    # # Gaussian kernel for SSIM calculation
    # def create_window(window_size):
    #     _1D_window = np.outer(np.hanning(window_size), np.hanning(window_size))
    #     return _1D_window / np.sum(_1D_window)
    #
    # window = create_window(window_size)

    # Calculate SSIM for each batch and band
    index = 0

    for j in range(bc):
        for i in range(bands):
            for uu in range(u):
                for vv in range(v):
                    if (uu == 0 and vv == 0) or (uu == 0 and vv == 4) or (uu == 4 and vv == 0) or (uu == 4 and vv == 4):
                        continue
                    # Extract the current band images
                    band_image1 = file1[j, i, uu, vv, :, :]
                    band_image2 = file2[j, i, uu, vv, :, :]
                    ssim_value = compare_ssim(band_image1, band_image2)
                    ssim_values[index] = ssim_value.item()
                    index += 1

    # Calculate and return the average SSIM
    ssim_values_nonzero = ssim_values[ssim_values != 0]
    average_ssim = np.mean(ssim_values_nonzero) if len(ssim_values_nonzero) > 0 else float('nan')

    # average_ssim = np.mean(ssim_values)
    return average_ssim

def ssim1(img1, img2, window, val_range):
    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2

    # Apply the Gaussian filter
    mu1 = scipy.ndimage.convolve(img1, window, mode='reflect')
    mu2 = scipy.ndimage.convolve(img2, window, mode='reflect')

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = scipy.ndimage.convolve(img1 ** 2, window, mode='reflect') - mu1_sq
    sigma2_sq = scipy.ndimage.convolve(img2 ** 2, window, mode='reflect') - mu2_sq
    sigma12 = scipy.ndimage.convolve(img1 * img2, window, mode='reflect') - mu1_mu2

    # Compute SSIM map
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)
def ssim_cal_val(file1, file2, window_size=7, val_range=1.0):
    """
    Calculate SSIM for each band and batch, then return the average SSIM.

    :param file1: The first input tensor (e.g., predicted image) of shape (batch, bands, rows, cols).
    :param file2: The second input tensor (e.g., ground truth image) of the same shape as file1.
    :param window_size: The size of the Gaussian window used for SSIM calculation.
    :param val_range: The range of the image values (default is 1.0 for normalized images).
    :return: The average SSIM value across all batches and bands.
    """

    # Check if the dimensions of the input images are the same
    if file1.shape != file2.shape:
        raise ValueError("输入图像的尺寸必须相同")

    # Get image dimensions
    file1 = lightfield_views_torch(file1)
    file2 = lightfield_views_torch(file2)

    batch, bands, u, v, rows, cols = file1.shape

    # Create an array to store SSIM values for each band
    ssim_values = np.zeros(bands * batch*u*v)

    # Gaussian kernel for SSIM calculation
    def create_window(window_size, channel):
        _1D_window = torch.Tensor(np.outer(np.hanning(window_size), np.hanning(window_size)))
        _2D_window = _1D_window.unsqueeze(0).unsqueeze(0)
        # _1D_window /= _1D_window.sum()
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    window = create_window(window_size, 1)

    # Calculate SSIM for each batch and band
    index = 0
    for j in range(batch):
        for i in range(bands):
            for uu in range (u):
                for vv in range (v):
                  if (uu == 0 and vv == 0) or (uu == 0 and vv == 4) or (uu == 4 and vv == 0) or (uu == 4 and vv == 4):
                    continue
            # Extract the current band images
                  band_image1 = file1[j, i,uu,vv,:, :]
                  band_image2 = file2[j, i,uu,vv,:, :]
                  ssim_value = ssim(band_image1, band_image2, window, window_size, val_range)
                  ssim_values[index] = ssim_value.item()
                  index += 1

    # Calculate and return the average SSIM
    ssim_values_nonzero = ssim_values[ssim_values != 0]
    average_ssim = np.mean(ssim_values_nonzero) if len(ssim_values_nonzero) > 0 else float('nan')
    # average_ssim = np.mean(ssim_values)
    return average_ssim
def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)
def to_lightfield(out_cube):
    ch,bs,h,w = out_cube.shape
    an = math.sqrt(bs)
    out_cube = out_cube.reshape(ch,int(an),int(an),h,w)
    # an = math.sqrt(ch)
    # out_cube = out_cube.reshape(bs,int(an),int(an),h,w)
    C,x,y,u,v, = out_cube.shape
    light2D = torch.zeros((x * u, y * v, C), dtype=out_cube.dtype,device=out_cube.device)

    # 使用嵌套循环来填充 light2D 数组
    for uu in range(x):
        for vv in range(y):
            for c in range(C):
                # 使用 NumPy 的高级索引来赋值
                light2D[uu::x,vv::y,c] = out_cube[c,uu, vv, :, :]


    return light2D
# def to_lightfield(out_cube):#输出的是HWC
#
#     C,x,y,u,v, = out_cube.shape
#     light2D = np.zeros((x * u, y * v, C), dtype=out_cube.dtype)
#
#     # 使用嵌套循环来填充 light2D 数组
#     for uu in range(x):
#         for vv in range(y):
#             for c in range(C):
#                 # 使用 NumPy 的高级索引来赋值
#                 light2D[uu::x,vv::y,c] = out_cube[c,uu, vv, :, :]
#
#     return light2D

def rearrange_view(image):
    """
    将形状为 (9H, 9W, c) 的光场图像重新排列，
    得到一个形状为 (9, 9, h, w, c) 的大图，表示每个视角的图像。

    参数：
    - image: 输入图像，形状为 (9H, 9W, c)
    - h: 每个视角图像的高度
    - w: 每个视角图像的宽度
    - c: 通道数

    返回：
    - rearranged_image: 重新排列后的大图，形状为 (9, 9, h, w, c)
    """
    # 初始化最终拼接后的大图，形状为 (9, 9, h, w, c)
    h,w,c = image.shape
    rearranged_image = np.zeros((9, 9, 36, 36, c), dtype=image.dtype)

    # 遍历9x9的视角网格，将每个视角的子图拼接到大图中
    for i in range(9):
        for j in range(9):
            # 提取每个视角的图像，保留逻辑不变
            sub_image = image[i::9, j::9, :]  # 每9个像素取一次，得到对应视角的图像

            # 确保提取的子图形状符合 (h, w, c

            # 将提取的子图粘贴到 9x9 视角网格的大图中
            rearranged_image[i, j, :, :, :] = sub_image

    return rearranged_image
def angle_resize(hyper,angle_number):
    u, v, H, W, C = hyper.shape
    new_data = np.zeros((angle_number, angle_number, H, W, C), dtype=hyper.dtype)
    if angle_number == 15:
        new_data = hyper
    else:
        step = 14 // (angle_number - 1)
        for i in range(angle_number):
            for j in range(angle_number):
                new_data[i, j] = hyper[i * step, j * step]
    new_data = new_data.transpose(4, 0, 1, 2, 3)
    out_data = to_lightfield(new_data)
    return out_data

def angle_supperresolution(out_data,angle_number):
    nh, nw, C = out_data.shape
    new_height = int(9 / 3.* nh)############这里的数字要改成角度分辨率
    new_width = int(9 / 3. * nw)
    resized_data = np.zeros((new_width, new_height, C), dtype=out_data.dtype)
    if angle_number == 9:
        return out_data
    else:
        for i in range(C):
            # 使用 cv2.resize 进行图像缩放
            resized_data[:, :, i] = cv2.resize(out_data[:, :, i], (new_width, new_height),
                                               interpolation=cv2.INTER_LINEAR)
        # resized_data = rearrange_view(resized_data)
        return resized_data
def Fast_rFFT2d_GPU_batch(DHSI, psf_cube, device):
    '''
    Args:
        DHSI: [bs, C, H, W]   通道 视角总数
        psf_cube: [bs, C, h, w]
    Returns:
        output_cube: [bs, C, H, W]
    not crop
    '''
    # DHSI = DHSI.astype(np.float32)
    DHSI = DHSI.transpose(1, 0, 2, 3)
    psf_cube = psf_cube.transpose(1, 0, 2, 3)
    DHSI = DHSI.astype(np.float32)  # 将数据类型转换为 float32
    psf_cube = psf_cube.astype(np.float32)
    DHSI = torch.from_numpy(DHSI)
    psf_cube = torch.from_numpy(psf_cube)
    DHSI = DHSI.to(device)
    psf_cube = psf_cube.to(device)
    [bs,C,H,W] = DHSI.shape      #36 25 430 550
    [bs,C,h,w] = psf_cube.shape

    output_cube = torch.zeros([bs, C, H, W], device=DHSI.device)
    # output_cube = torch.zeros([bs, C, H, W])
    # prepare padding size to accommodate FFT-based convolution
    YH = H + h - 1
    YW = W + w - 1
    YH = 2 ** (int(torch.log2(torch.tensor(YH)))+1)
    YW = 2 ** (int(torch.log2(torch.tensor(YW)))+1)
    for i in range(bs):
        HSI = DHSI[i]   # [C,H,W]
        psf = psf_cube[i]   # [C,h,w]
        HSI_padded = torch.zeros([C, YH, YW], device=HSI.device)
        psf_padded = torch.zeros([C, YH, YW], device=psf.device)
        HSI_padded[:, (YH - H) // 2:(YH - H) // 2 + H, (YW - W) // 2:(YW - W) // 2 + W] = HSI
        psf_padded[:, (YH - h) // 2:(YH - h) // 2 + h, (YW - w) // 2:(YW - w) // 2 + w] = psf
        HSI_fft = torch.fft.rfft2(HSI_padded, s=(YH,YW)).cuda()
        psf_fft = torch.fft.rfft2(psf_padded, s=(YH,YW)).cuda()
        # Element-wise multiplication in the frequency domain
        conv_result_fft = HSI_fft * psf_fft
        # inverse FFT to get the convolution result in the spatial domain
        conv_result = torch.fft.irfft2(conv_result_fft)[:,:YH,:YW]
        # roll
        start_y = (YH - H) // 2 -1
        start_x = (YW - W) // 2 -1
        conv_result = torch.roll(conv_result, shifts=(YH//2, YW//2), dims=(1,2))
        meas = conv_result[:, start_y:start_y + H, start_x:start_x + W]
        # crop the result to the valid region
        # meas = conv_result[:, YH//2-H//2:YH//2+H//2, YW//2-W//2:YW//2+W//2]
        # output_cube[i] = meas.squeeze(0)
        output_cube[i] = meas

    del HSI_padded, psf_padded, HSI_fft, psf_fft, conv_result_fft, conv_result,DHSI
    torch.cuda.empty_cache()
    # result = output_cube.cpu().numpy()

    # result = result.reshape(bs,u,v,H,W)
    result = output_cube
    del output_cube
    torch.cuda.empty_cache()
    # result = to_lightfield(result)
    return result


def lightfield_resize(out_cube, alpha):
    # light2D_cpu = out_cube.cpu()  # 将张量移到 CPU
    # out_cube = out_cube.permute(1, 0, 2, 3)
    light2D_cpu = out_cube  #
    ch, bs, h, w = light2D_cpu.shape
    an = math.sqrt(bs)
    new_H = h / int(an)
    new_W = w / int(an)
    new_H = int(new_H) * alpha
    new_W = int(new_W) * alpha
    resized_light_field = torch.zeros(ch, bs, new_H, new_W,device=light2D_cpu.device)

    # 对每个视角进行 resize
    for i in range(121):
        # 获取当前视角的图像
        view_image = light2D_cpu[:, i, :, :].unsqueeze(0)  # shape [1, 3, H, W] 适应插值
        # 调整大小
        resized_view = F.interpolate(view_image, size=(new_H, new_W), mode='bicubic', align_corners=False)
        # 存储到结果张量中
        resized_light_field[:, i, :, :] = resized_view.squeeze(0)
    return resized_light_field  # shape [3, new_H, new_W]
    # rgb = hyperspectral_to_rggb(result)
    # return rgb
def lightfield_resize_train(out_cube, alpha):
    # light2D_cpu = out_cube.cpu()  # 将张量移到 CPU
    # out_cube = out_cube.permute(1, 0, 2, 3)
    light2D_cpu = out_cube  #
    c,ch, bs, h, w = light2D_cpu.shape
    an = math.sqrt(bs)
    new_H = h / int(an)
    new_W = w / int(an)
    new_H = int(new_H) * alpha
    new_W = int(new_W) * alpha
    resized_light_field = torch.zeros(c,ch, bs, new_H, new_W,device=light2D_cpu.device)

    # 对每个视角进行 resize
    for j in range(c):
        for i in range(121):
            # 获取当前视角的图像
            view_image = light2D_cpu[j,:, i, :, :].unsqueeze(0)  # shape [1, 3, H, W] 适应插值
            # 调整大小
            resized_view = F.interpolate(view_image, size=(new_H, new_W), mode='bicubic', align_corners=False)
            # 存储到结果张量中
            resized_light_field[j,:, i, :, :] = resized_view.squeeze(0)
    return resized_light_field

def RGB_para(temp):
    """
    Returns the filter response array for the specified color channel.

    Parameters:
    - temp (str): The color channel to retrieve ('R', 'G', or 'B').

    Returns:
    - np.ndarray: A NumPy array of filter responses for the specified channel.

    Raises:
    - ValueError: If an invalid channel is specified.
    """

    if temp == 'R':
        para = np.array([
            0.01742, 0.01657986, 0.01547497 ,0.01450286, 0.01356173 ,0.0125952,
         0.01425043, 0.02205597, 0.03279895, 0.04388674 ,0.04919325, 0.04139934,
         0.02973248, 0.02211241, 0.0353109,  0.09137389, 0.39352125, 0.53584973,
         0.61930207, 0.63292726, 0.61844552, 0.60103229, 0.58661661, 0.56988353,
         0.55010855, 0.52568045, 0.47178398, 0.36211826, 0.1504242,  0.09398859,
         0.04508742, 0.0216653,  0.01270103, 0.00623589 ,0.00282291, 0.00192292
        ], dtype=np.float32)
    elif temp == 'G':
        para = np.array([
            0.05121, 0.04969485, 0.07954201, 0.17634375, 0.36545909, 0.50806916,
         0.58545936, 0.62158448, 0.64362692, 0.66761213, 0.67000457 ,0.66575176,
         0.63816276, 0.59655744, 0.56483997, 0.53828899, 0.49677735, 0.44964133,
         0.39044315, 0.27499602, 0.20688836, 0.16114985, 0.13215223, 0.10716299,
         0.09247162, 0.0903454,  0.09098597, 0.08415912, 0.05648338, 0.0313006,
         0.01712269, 0.01127559 ,0.00789912, 0.004998,  0.00261123 ,0.00095196
        ], dtype=np.float32)
    elif temp == 'B':
        para = np.array([
            0.53125, 0.53506533, 0.53187696, 0.51383448, 0.48327002, 0.42712029,
         0.34655248, 0.2838295,  0.19628517, 0.14450326, 0.10975592, 0.08086249,
         0.06392193, 0.046838,   0.03421378, 0.02603674, 0.02191885, 0.02064777,
         0.01853797, 0.01589715, 0.01302589, 0.01121186, 0.01322195, 0.02255729,
         0.03044202, 0.03281034, 0.03347472, 0.03153852, 0.02311025, 0.00970481,
         0.00642677, 0.00670284, 0.00585643, 0.00484767, 0.00390575, 0.00319862
        ], dtype=np.float32)
    else:
        raise ValueError("Invalid channel selected. Choose 'R', 'G', or 'B'.")

    return torch.from_numpy(para)
def hyperspectral_to_rggb(hyperspectral_data):
    # 假设 RGB_para 已被修改为在 PyTorch 上兼容的函数
    # R_curve = torch.tensor(RGB_para('R'), dtype=torch.float32).to(hyperspectral_data.device)
    # G_curve = torch.tensor(RGB_para('G'), dtype=torch.float32).to(hyperspectral_data.device)
    # B_curve = torch.tensor(RGB_para('B'), dtype=torch.float32).to(hyperspectral_data.device)
    R_curve = RGB_para('R').clone().detach().to(hyperspectral_data.device).float()
    G_curve = RGB_para('G').clone().detach().to(hyperspectral_data.device).float()
    B_curve = RGB_para('B').clone().detach().to(hyperspectral_data.device).float()

    # 初始化 RGB 图像
    rows, cols, _ = hyperspectral_data.shape
    rgb_image = torch.zeros((3, rows, cols), dtype=torch.float32, device=hyperspectral_data.device)

    # 计算 R, G, B 通道
    R_channel = torch.sum(hyperspectral_data * R_curve.view(1, 1, -1), dim=-1)
    G_channel = torch.sum(hyperspectral_data * G_curve.view(1, 1, -1), dim=-1)
    B_channel = torch.sum(hyperspectral_data * B_curve.view(1, 1, -1), dim=-1)

    rgb_image[0, :, :] = R_channel
    rgb_image[1, :, :] = G_channel
    rgb_image[2, :, :] = B_channel

    del R_channel ,G_channel,B_channel
    torch.cuda.empty_cache()

    return rgb_image
def hyperspectral_to_rgb(hyperspectral_data):
    # 获取响应曲线
    R_curve = RGB_para('R')
    G_curve = RGB_para('G')
    B_curve = RGB_para('B')

    # 初始化RGGB马赛克图像
    rows, cols, _ = hyperspectral_data.shape
    # rggb_image = np.zeros((2*rows, 2*cols), dtype=np.float32)

    rgb_image = np.zeros((3,rows,cols), dtype=np.float32)
    # 初始化滤波参数矩阵
    # filter_params = np.zeros((rows, cols, 36), dtype=np.float32)
    # Compute the R, G, B channels
    R_channel = np.sum(hyperspectral_data * R_curve[None, None, :], axis=-1)
    G_channel = np.sum(hyperspectral_data * G_curve[None, None, :], axis=-1)
    B_channel = np.sum(hyperspectral_data * B_curve[None, None, :], axis=-1)

    rgb_image[0,:, :] = R_channel
    rgb_image[1,:, :] = G_channel
    rgb_image[2,:, :] = B_channel

    # rggb_image[0::2, 0::2] = R_channel  # R
    # rggb_image[0::2, 1::2] = G_channel  # G1
    # rggb_image[1::2, 0::2] = G_channel  # G2
    # rggb_image[1::2, 1::2] = B_channel
    return rgb_image


def rearrange_light_field(image):
    """
    将形状为 (9H, 9W, c) 的光场图像重新排列，
    得到一个拼接后的大图，每个视角的子图排列在一起。

    参数：
    - image: 输入图像，形状为 (9H, 9W, c)
    - H: 每个视角图像的高度
    - W: 每个视角图像的宽度
    - c: 通道数

    返回：
    - rearranged_image: 重新排列后的大图，形状为 (9H, 9W, c)
    c,H,
    """
    c,H,W = image.shape
    # 初始化最终拼接后的大图形状
    rearranged_image = np.zeros((c, H,  W, ), dtype=image.dtype)

    # 遍历9x9的视角网格，将每个视角的子图拼接到大图中
    for i in range(9):
        for j in range(9):
            # 提取每个视角的图像
            sub_image = image[:,i::9, j::9,]  # 每9个像素取一次，得到对应视角的图像

            # 将提取的子图粘贴到大图的适当位置
            rearranged_image[:,i * 36:(i + 1) * 36, j *36:(j + 1) * 36] = sub_image

    return rearranged_image

def torch_sam(img, ref):  # input [28,256,256]
    #  GitHub - MyuLi/SERT
    sum1 = torch.sum(ref * img, 0)
    sum2 = torch.sum(ref * ref, 0)
    sum3 = torch.sum(img * img, 0)
    t = (sum2 * sum3) ** 0.5
    numlocal = torch.gt(t, 0)
    num = torch.sum(numlocal)
    t = sum1 / t
    angle = torch.acos(t)
    sumangle = torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle).sum()
    if num == 0:
        averangle = sumangle
    else:
        averangle = sumangle / num
    SAM = averangle * 180 / 3.14159256
    return SAM
def lightfield_resize_out(out_cube, alpha):
    """
    调整光场数据的尺寸，支持批次大小。

    参数:
    - out_cube (torch.Tensor): 输入张量，形状为 (batch_size, bs, ch, h, w)。
    - alpha (int): 调整大小的比例因子。

    返回:
    - resized_light_field (torch.Tensor): 调整后的光场数据，形状为 (batch_size, bs, ch, new_H, new_W)。
    """
    out_cube = out_cube.permute(0,2,3,4,5,1)
    batch_size, ch,ch, h, w,bs = out_cube.shape
    data_reshaped = out_cube.reshape(batch_size,  121, h, w,bs)
    an = ch


    # 计算新高度和宽度
    new_H = int((h // an) * alpha)
    new_W = int((w // an) * alpha)

    # 初始化结果张量
    resized_light_field = torch.zeros(batch_size, 121,  bs,new_H, new_W, device=out_cube.device)

    # 对每个批次和每个视角进行 resize
    for b in range(batch_size):
        for i in range(121):
            # 获取当前批次的当前视角图像
            view_image = data_reshaped[b, i]  # shape [ch, h, w]
            # 调整大小
            view_image = view_image.permute(2,0,1)
            resized_view = F.interpolate(view_image.unsqueeze(0), size=(new_H, new_W), mode='bicubic', align_corners=False)
            # 存储到结果张量中
            resized_light_field[b, i] = resized_view.squeeze(0)

    return resized_light_field  # shape [batch_size, bs, ch, new_H, new_W]

def to_view(stitched_data, u=11, v=11):
    """
    将拼接后的数据转换为光场格式，支持批次大小。

    参数:
    - stitched_data (torch.Tensor): 拼接后的数据，形状为 (batch_size, x, y, C)。
    - u (int): 光场在第一个维度的位置数量。
    - v (int): 光场在第二个维度的位置数量。

    返回:
    - light2D (torch.Tensor): 光场数据，形状为 (batch_size, C, u, v, x/u, y/v)。
    """
    # 确保输入数据是一个张量
    if not isinstance(stitched_data, torch.Tensor):
        raise TypeError("输入数据必须是一个 PyTorch 张量")

    batch_size,C, x, y,  = stitched_data.shape
    h = x // u  # 每个子视图的高度
    w = y // v  # 每个子视图的宽度

    # 检查 x 和 y 是否能被 u 和 v 整除
    if x % u != 0 or y % v != 0:
        pad_x = (u - x % u) % u
        pad_y = (v - y % v) % v

        stitched_data = torch.nn.functional.pad(
            stitched_data,
            (0, pad_y, 0, pad_x),  # (left, right, top, bottom) 注意顺序和你用的框架
            mode='constant', value=0
        )
        h = (x + pad_x) // u
        w = (y + pad_y) // v
        # raise ValueError("x 和 y 必须能被 u 和 v 整除")

    # 初始化 light2D 张量，保持与 stitched_data 相同的数据类型和设备
    light2D = torch.zeros((batch_size, C, u, v, h, w), dtype=stitched_data.dtype, device=stitched_data.device)

    # 转置 stitched_data，使通道维度 C 排在第二位
    # stitched_data = stitched_data.permute(0, 3, 1, 2)  # 变成 (batch_size, C, x, y)

    # 填充 light2D 张量
    for uu in range(u):
        for vv in range(v):
            # 提取子视图并赋值到 light2D
            light2D[:, :, uu, vv, :, :] = stitched_data[:, :, uu::u, vv::v]

    return light2D

def to_lightfield_out(out_cube):
    """
    将输入张量转换为光场格式，支持批次大小。

    参数:
    - out_cube (torch.Tensor): 输入张量，形状为 (batch_size, ch, bs, h, w)。

    返回:
    - light2D (torch.Tensor): 光场数据，形状为 (batch_size, x * u, y * v, C)。
    """
    out_cube = out_cube.permute(0,2,1,3,4)
    batch_size, ch, bs, h, w = out_cube.shape
    an = int(math.sqrt(bs))

    if an * an != bs:
        raise ValueError("视角数量必须是一个完全平方数。")

    # 重塑为 (batch_size, ch, an, an, h, w)
    out_cube = out_cube.reshape(batch_size, ch, an, an, h, w)

    # 初始化 light2D 张量
    light2D = torch.zeros((batch_size, h * an, w * an, ch), dtype=out_cube.dtype, device=out_cube.device)

    # 使用嵌套循环来填充 light2D 数组
    for b in range(batch_size):
        for uu in range(an):
            for vv in range(an):
                for c in range(ch):
                    # 将视角图像逐一放入光场网格
                    light2D[b, uu::an, vv::an, c] = out_cube[b, c, uu, vv, :, :]

    return light2D

def cal_metrics(img1, img2):
    # if len(img1.size())==2:
    #     [H, W] = img1.size()
    #     img1 = img1.view(angRes, H // angRes, angRes, W // angRes).permute(0,2,1,3)
    # if len(img2.size())==2:
    #     [H, W] = img2.size()
    #     img2 = img2.view(angRes, H // angRes, angRes, W // angRes).permute(0,2,1,3)

    bs , h, w ,c= img1.shape

    PSNR = np.zeros(bs * c)
    SSIM = np.zeros(bs * c)

    idex = 0
    for j in range(bs):
        for i in range(c):
            PSNR[idex] = cal_psnr(img1[j,  :, :,i], img2[j, :, :,i])
            SSIM[idex] = cal_ssim(img1[j, :, :,i], img2[j,  :, :,i])
            idex =idex +1


    psnr_mean = PSNR.sum() / np.sum(PSNR > 0)
    ssim_mean = SSIM.sum() / np.sum(SSIM > 0)

    return psnr_mean, ssim_mean

def cal_psnr(img1, img2,):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()

    return metrics.peak_signal_noise_ratio(img1_np, img2_np ,data_range=1.0)


def cal_ssim(img1, img2):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()
    return cal_ssim_cv2(img1_np, img2_np)
    # return metrics.structural_similarity(img1_np, img2_np, data_range=1.0,gaussian_weights=True)
def cal_ssim_cv2(img1, img2, kernel_size=15, sigma=2, K1=0.03, K2=0.05, data_range=1.0):
    """img1, img2: float32 np.array, shape (H, W), range [0,1]"""
    img1 = np.clip(img1.astype(np.float32), 0, 1)
    img2 = np.clip(img2.astype(np.float32), 0, 1)

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = cv2.GaussianBlur(img1, (kernel_size, kernel_size), sigma)
    mu2 = cv2.GaussianBlur(img2, (kernel_size, kernel_size), sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (kernel_size, kernel_size), sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (kernel_size, kernel_size), sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (kernel_size, kernel_size), sigma) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


# def lightfield_angluarresize(out_cube,alpha):
#     # light2D_cpu = out_cube.cpu()  # 将张量移到 CPU
#     out_cube = out_cube.permute(1,0,2,3)
#     light2D_cpu = out_cube#
#     ch,bs,h,w = light2D_cpu.shape
#     light2D = light2D_cpu.reshape(11,11,bs,h,w)
#     light2D = light2D.permute(2,3,4,0,1,)
#     light2D = light2D.reshape(bs,h*w,11,11)
#     resized_light_field = F.interpolate(light2D, size=(alpha, alpha), mode='bicubic', align_corners=False)
#     resized_light_field =resized_light_field.reshape(bs,h,w,alpha*alpha)
#     return resized_light_field
def lightfield_angluarresize(out_cube,alpha):
    # light2D_cpu = out_cube.cpu()  # 将张量移到 CPU
    out_cube = out_cube.permute(1,0,2,3)
    light2D_cpu = out_cube#
    ch,bs,h,w = light2D_cpu.shape
    light2D = light2D_cpu.reshape(11,11,bs,h,w)
    light2D = light2D.permute(2,3,4,0,1,)
    light2D = light2D.reshape(bs,h*w,11,11)
    resized_light_field = F.interpolate(light2D, size=(alpha, alpha), mode='bicubic', align_corners=False)
    resized_light_field =resized_light_field.reshape(bs,h,w,alpha*alpha)
    return resized_light_field