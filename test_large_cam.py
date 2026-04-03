"""测试"""
from config import *
import os
import time
import torch
from My_dataloader_3s_large import Data_loader
from model import network, network_pp, network_pp_ppm, network_pp_sp_ftt, network_pp_sp, TransformerWithGCN
from model_large import Samunet_Segmentation_Model
from model_sam.build_sam import sam_model_registry
from runner.metrics import calculate_metrics1
import numpy as np
import random
import csv
import matplotlib.pyplot as plt
from visualize import visualization_segmentation as vis
import warnings
warnings.filterwarnings("ignore")

""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
SAVE_tes_ROOT_heatmap= r'.\summary\com_3tranSUNet_fold4_epoch\attp'
SAVE_tes_ROOT_cam= r'.\summary\com_3tranSUNet_fold4_epoch\cam'
os.makedirs(SAVE_tes_ROOT_heatmap, exist_ok=True)
os.makedirs(SAVE_tes_ROOT_cam, exist_ok=True)


def save_overlay(image, cam, filename, alpha=0.5):
    """
    生成并保存叠加可视化图
    :param image: 原始图像 [H, W] 或 [H, W, 3]
    :param cam: 热力图 [H, W]
    :param filename: 保存文件名（不含后缀）
    :param alpha: 热力图透明度 (0-1)
    """
    # 数据预处理
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.ndim == 3 and image.shape[0] == 1:  # 处理单通道图像
        image = image.squeeze()

    # 归一化图像
    if image.dtype == np.float32:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

    # 创建彩色热力图
    heatmap = plt.get_cmap('jet')(cam)[..., :3]  # [H, W, 3], 值范围[0,1]
    heatmap = (heatmap * 255).astype(np.uint8)

    # 转换原始图像为RGB（兼容灰度图）
    if image.ndim == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image

    # 调整对比度（可选）
    image_rgb = np.clip(image_rgb * 1.2, 0, 255).astype(np.uint8)

    # 叠加显示
    overlay = (image_rgb * (1 - alpha) + heatmap * alpha).astype(np.uint8)

    # 创建对比视图
    plt.figure(figsize=(18, 6))

    # 子图1：原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb if image_rgb.shape[-1] == 3 else image,
               cmap='gray' if image.ndim == 2 else None)
    plt.title('Original Image')
    plt.axis('off')

    # 子图2：热力图
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title('Activation Heatmap')
    plt.axis('off')

    # 子图3：叠加效果
    plt.subplot(1, 3, 3)
    plt.imshow(image_rgb if image_rgb.shape[-1] == 3 else image,
               cmap='gray' if image.ndim == 2 else None)
    plt.imshow(heatmap, alpha=alpha)
    plt.title('Overlay (alpha={})'.format(alpha))
    plt.axis('off')

    # 保存图像
    plt.savefig(
        os.path.join(SAVE_tes_ROOT_cam, f'{filename}_overlay.png'),
        bbox_inches='tight',
        pad_inches=0,
        dpi=150
    )
    plt.close()
def save_heatmap(cam, filename):
    """
    保存纯热力图
    cam: 激活图 [H, W]
    filename: 保存文件名
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(cam, cmap='jet')
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(
        os.path.join(SAVE_tes_ROOT_heatmap, f'{filename}.png'),
        bbox_inches='tight',
        pad_inches=0,
        dpi=300  # 提高输出分辨率
    )
    plt.close()

from torch import nn
# 新增GradCAM相关工具函数
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册钩子
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def get_cam(self, input_image, image, mask, target_category=None):
        # 前向传播
        sam_vector, foreground_vector, background_vector, pred, e_list = self.model(input_image, image, compute_sam_feat=False)
        # 生成目标梯度（保持4D形状）
        target_mask = torch.sigmoid(pred) .float()  # [B, C, H, W]

        # 创建梯度张量（关键修改）
        gradient = torch.ones_like(pred) * mask

        # 反向传播
        self.model.zero_grad()
        pred.backward(gradient=gradient, retain_graph=True)

        # 计算权重
        grad_mean = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        grad_std = torch.std(self.gradients, dim=[2, 3], keepdim=True)
        pooled_gradients = grad_mean / (grad_std + 1e-7)  # 标准化处理

        # 生成CAM
        cam = torch.mul(self.activations, pooled_gradients).sum(dim=1, keepdim=True)
        cam = nn.functional.relu(cam)

        cam = nn.functional.interpolate(cam, size=input_image.shape[2:], mode='bilinear')
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam.squeeze().cpu().numpy()


if __name__ == "__main__":
    """ Seeding """
    seeding(123)
    os.makedirs(SAVE_tes_ROOT_catimage, exist_ok=True)
    os.makedirs(SAVE_tes_ROOT_img, exist_ok=True)
    os.makedirs(SAVE_tes_ROOT_mask, exist_ok=True)
    os.makedirs(SAVE_tes_ROOT_pred, exist_ok=True)
    test_set = Data_loader(data_ROOT, num=4, test=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size_test, shuffle=False)

    # 模型文件夹路径
    models_dir = r"D:\xuchao\peat\peat_task1\code_3SUNet_transunet_0.2_3_fold4_patch1_TAM\summary\com_3tranSUNet_fold4_epoch\ckpt\best148.pth"

    # 创建保存结果的 CSV 文件
    result_csv_path = SAVE_ROOT + "test/test.csv"

    with open(result_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        """ Load and test each model """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        time_taken = []
        metrics_dict = {'CAL': [], 'MET': []}
        image_name = []
        file1 = []
        file2 = []
        file3 = []
        file4 = []

        # 创建并加载模型
        sam = sam_model_registry["vit_b"]().to('cpu')
        model = Samunet_Segmentation_Model(sam, 1, 1).to(device)
        # model = network().to(device)
        checkpoint = torch.load(models_dir, map_location=device)
        # 清除不需要的键
        if 'net' in checkpoint:
            model.load_state_dict(checkpoint['net'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        # target_layer = model.decoder.Up_conv2.conv[-3]   # 示例路径，需根据实际模型调整
        target_layer = model.decoder.Conv   # 示例路径，需根据实际模型调整

        # 初始化GradCAM
        gradcam = GradCAM(model, target_layer)

        for step, samples in enumerate(test_loader):
            input_image, image, mask, file_name = samples
            file_name = file_name[0].replace('.png', '')
            start_time = time.time()
            # 需要梯度计算
            input_image.requires_grad = True
            image.requires_grad = True

            # 生成GradCAM
            with torch.enable_grad():
                cam = gradcam.get_cam(input_image, image, mask)

            with torch.no_grad():
                sam_vector, foreground_vector, background_vector, pred, e_list = model(input_image, image, compute_sam_feat=False)

            # 热力图生成
            heatmaps = torch.sigmoid(pred).cpu().numpy()  # [B, C, H, W]

            for i in range(image.size(0)):
                # 保存热力图
                current_heatmap = heatmaps[i, 0, ...]  # 假设单通道输出
                save_heatmap(current_heatmap, f"{file_name}_{i}_heatmap")
            # 在测试循环中调用
            for i in range(image.size(0)):
                img = image[i, 0, ...].detach().cpu().numpy()  # 原始图像
                activation = cam  # GradCAM输出

                # 统一归一化到0-1
                img = (img - img.min()) / (img.max() - img.min())
                activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)

                save_overlay(img, activation, f"{file_name}_{i}", alpha=0.6)

            total_time = time.time() - start_time

            time_taken.append(total_time)
            image_name.append(file_name)

            metrics = calculate_metrics1(torch.sigmoid(pred.data).cpu().numpy(), mask.data.cpu().numpy())
            metrics_dict['CAL'].append(metrics['CAL']['value'])
            metrics_dict['MET'].append(metrics['MET']['value'])
            file1.append(metrics['MET']['value'][0])  # pre
            file2.append(metrics['MET']['value'][1])  # recall
            file3.append(metrics['MET']['value'][2])  # f1
            file4.append(metrics['MET']['value'][6])  # hf
            for i in range(image.size(0)):
                vis(image[i, 0, ...], torch.sigmoid(pred.data)[i, 0, ...], mask[i, 0, ...], '{}'.format(file_name), phase='test')

        average_time_taken = np.mean(time_taken[1:])

        cal = np.around(np.vstack(metrics_dict['CAL']).mean(axis=0), decimals=4)
        met = np.around(np.vstack(metrics_dict['MET']).mean(axis=0), decimals=4)
        met1 = np.around(np.vstack(metrics_dict['MET']).std(axis=0), decimals=4)
        mean_std = np.array([f"{mean}±{std}" for mean, std in zip(met, met1)])

        precision, recall, f1, acc, jaccard, dice, hf, name = mean_std[0], mean_std[1], mean_std[2], mean_std[3], mean_std[4], mean_std[5], mean_std[6], image_name[0]

        file1_mean = round(np.mean(file1), 3)
        file1_std = round(np.std(file1), 3)

        file2_mean = round(np.mean(file2), 3)
        file2_std = round(np.std(file2), 3)

        file3_mean = round(np.mean(file3), 3)
        file3_std = round(np.std(file3), 3)

        file4_mean = round(np.mean(file4), 3)
        file4_std = round(np.std(file4), 3)
        print("pre:" + str(file1_mean) + "±" + str(file1_std))
        print("re:" + str(file2_mean) + "±" + str(file2_std))
        print("di:" + str(file3_mean) + "±" + str(file3_std))
        print("hf:" + str(file4_mean) + "±" + str(file4_std))

        image_name = [name for name in image_name]
        data_to_write = list(zip(image_name, file1, file2, file3, file4))
        writer.writerows(data_to_write)
