import torch
import torch.nn as nn
from torchvision import models
# from segment_anything import sam_model_registry
from model_sam import sam
from model_sam.unet import UNet, UNetEncoder, UNetDecoder, UNetDecoderPlus
from torch.nn import functional as F
from vit import ViT
from einops import rearrange


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out



from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dense_to_sparse

class TopoAttentionModule(nn.Module):
    def __init__(self, in_channels=32, patch_size=16, threshold=0.5):
        """
        in_channels: 输入特征图的通道数
        patch_size: patch 的空间尺寸（例如16）
        threshold: 皮尔逊相关系数阈值，用于构建图的邻接矩阵
        """
        super(TopoAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.threshold = threshold

        # 定义两层 GATv2Conv
        # 输入节点特征维度为 in_channels, 多头数为8，每个头分得 in_channels/8，拼接后输出 in_channels
        self.gatv2_1 = GATv2Conv(in_channels=in_channels,
                                 out_channels=in_channels // 8,
                                 heads=8, concat=True, edge_dim=1)
        self.gatv2_2 = GATv2Conv(in_channels=in_channels,
                                 out_channels=in_channels // 8,
                                 heads=8, concat=True, edge_dim=1)

    def pearson_corr(self, x, eps=1e-8):
        """
        计算皮尔逊相关系数
        x: (B, N, C) 每个节点的特征向量（N: patch 数量, C: 特征维度）
        输出: (B, N, N) 的相关矩阵
        """
        # 对每个节点的特征向量做中心化
        x_centered = x - x.mean(dim=-1, keepdim=True)  # (B, N, C)
        # 计算节点间内积，得到 (B, N, N)
        numerator = torch.matmul(x_centered, x_centered.transpose(-1, -2))
        # 计算每个节点的范数 (B, N)
        norms = x_centered.norm(dim=-1)
        # 计算外积，避免除以0
        denom = torch.matmul(norms.unsqueeze(-1), norms.unsqueeze(-2)) + eps
        corr = numerator / denom
        return corr

    def forward(self, x):
        """
        x: 输入特征图，形状 (B, C, 256, 256)
        流程：
         1. Patch Partition： (B, C, 256,256) -> (B, C, num_patches, patch_size, patch_size)
         2. 展开每个 patch（但这里后续直接用均值）
         3. 构建图关系：计算皮尔逊相关系数 & 阈值化 -> 得到 (B, num_patches, num_patches) 邻接矩阵
         4. 节点特征：对每个 patch 做均值池化 -> (B, num_patches, C)
         5. 两层 GATv2 处理：更新节点特征
         6. 恢复空间：将 (B, num_patches, C) 重塑为 (B, C, 16,16) -> 上采样到 (B, C, 256,256) -> 与输入相加
        """
        B, C, H, W = x.shape  # 例如 (8, 32, 256, 256)
        ps = self.patch_size
        # 1. Patch Partition
        # 将 (B, C, H, W) reshape 成 (B, C, H//ps, ps, W//ps, ps)
        x_patches = x.reshape(B, C, H // ps, ps, W // ps, ps)
        # 调整为 (B, C, num_patches, ps, ps)
        num_patches = (H // ps) * (W // ps)  # 16*16=256
        x_patches = x_patches.permute(0, 2, 4, 1, 3, 5).reshape(B, num_patches, C, ps, ps)
        # 此时 x_patches 的 shape: (B, 256, 32, 16, 16)

        # 4. 节点特征：对每个 patch 做均值池化 (在 patch 内空间上求均值)
        # 计算均值后 shape: (B, 256, 32)
        node_features = x_patches.mean(dim=[-1, -2])  # 在 16x16 上做均值
        # 为后续 GATv2，保持形状 (B, num_patches, C)
        # （注：这里每个 patch 的特征为 32 维向量）

        # 2 & 3. 构建图关系：计算皮尔逊相关系数 & 阈值化
        # 先计算相关矩阵：输入形状 (B, 256, 32) -> 先转为 (B, 256, C)
        corr_matrix = self.pearson_corr(node_features)  # (B, 256, 256)
        # 阈值化：大于 threshold 置 1，其余置 0，得到邻接矩阵
        adj_matrix = (corr_matrix > self.threshold).float()  # (B, 256, 256)

        # 为了使用 PyG 的 GATv2Conv，需要构建稀疏图
        # 将每个样本的稠密邻接矩阵转为 edge_index 与 edge_weight
        edge_index_list = []
        edge_weight_list = []
        batch_index_list = []
        N = num_patches  # 256
        for b in range(B):
            # adj_matrix[b]: (256,256)
            edge_index, edge_weight = dense_to_sparse(adj_matrix[b])
            # edge_index 中的节点索引为 0~255, 为了构造成批图，我们将其偏移
            edge_index = edge_index + b * N
            edge_index_list.append(edge_index)
            edge_weight_list.append(edge_weight)
            batch_index_list.append(torch.full((N,), b, device=x.device, dtype=torch.long))
        edge_index = torch.cat(edge_index_list, dim=1)   # shape (2, total_edges)
        edge_weight = torch.cat(edge_weight_list, dim=0)   # shape (total_edges,)
        batch_vector = torch.cat(batch_index_list, dim=0)  # (B*N,)

        # 将 node_features 重塑为 (B*N, C)
        node_features_flat = node_features.reshape(B * N, C)

        # 5. 两层 GATv2 处理
        # 第一层 GATv2Conv: 输入 (B*N, C) -> 输出 (B*N, C)（内部采用8头，每头输出 C/8）
        out = self.gatv2_1(node_features_flat, edge_index, edge_weight)
        out = F.elu(out)
        # 第二层
        out = self.gatv2_2(out, edge_index, edge_weight)
        out = F.elu(out)
        # 恢复形状为 (B, num_patches, C)
        out = out.reshape(B, N, C)

        # 6. 空间恢复
        # 将 (B, 256, C) 重塑为 (B, 16, 16, C)
        grid_features = out.reshape(B, H // ps, W // ps, C)
        # 调整维度为 (B, C, 16, 16)
        grid_features = grid_features.permute(0, 3, 1, 2)
        # 上采样到 (B, C, 256, 256)
        # restored_features = F.interpolate(grid_features, size=(H, W), mode='bilinear', align_corners=False)
        restored_features = grid_features.repeat_interleave(H//16, dim=2).repeat_interleave(W//16, dim=3)  # (B, C, 256, 256)
        # 残差连接：与原始特征图相加
        out_final = restored_features + x  # (B, C, 256, 256)

        # shape 流程说明：
        # x: (B, 32, 256, 256)
        # -> x_patches: (B, 256, 32, 16, 16)
        # -> node_features (均值池化): (B, 256, 32)
        # -> 相关矩阵: (B, 256, 256) 经过阈值化后构成邻接矩阵
        # -> GATv2 输入: node_features_flat: (B*256, 32)
        # -> GATv2 输出: (B*256, 32) -> 重塑为 (B, 256, 32)
        # -> 重塑为网格: (B, 16, 16, 32) -> (B, 32, 16, 16)
        # -> 上采样: (B, 32, 256, 256)
        # -> 与原始 x 相加: (B, 32, 256, 256)
        return out_final









# 定义一个用于下游任务（例如分割任务）的模型
class Samunet_Segmentation_Model(nn.Module):
    def __init__(self, sam_model, in_chs, out_chs):
        super(Samunet_Segmentation_Model, self).__init__()
        self.samEncoder_model = sam_model.image_encoder
        self.encoder = UNetEncoder(in_chs)

        self.vit = ViT(img_dim=16,
              in_channels=512,
              patch_dim=1,
              embedding_dim=512,
              block_num=8,
              head_num=4,
              mlp_dim=512)

        self.fc_b = nn.Conv2d(512,256, 1)
        self.fc_f = nn.Conv2d(512,256, 1)
        # self.fc_b = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        # self.fc_f = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.decoder = UNetDecoder(out_chs)
        # 仅在跨层连接的部分添加 CBAM
        # self.cbam_e1 = CBAM(32)  # 适配 e1
        # self.cbam_e2 = CBAM(64)  # 适配 e2
        # self.cbam_e3 = CBAM(128)  # 适配 e3
        # self.cbam_e4 = CBAM(256)  # 适配 e4
        # self.cbam_e5 = CBAM(256)  # 适配 e5

        # 仅在跨层连接的部分添加 TPAM
        self.TPAM_e1 = TopoAttentionModule(32, 16)  # 适配 e1
        self.TPAM_e2 = TopoAttentionModule(64, 8)  # 适配 e2
        self.TPAM_e3 = TopoAttentionModule(128, 4)  # 适配 e3
        self.TPAM_e4 = TopoAttentionModule(256, 2)  # 适配 e4



        self.decoder = UNetDecoder(out_chs)
        # 用于存储梯度
        self.gradients = {}


    def save_gradient(self, module_name):
        """hook函数，存储特定层的梯度"""

        def hook(grad_output):
            self.gradients[module_name] = grad_output  # 只存储输出的梯度

        return hook

    def forward(self, in_x, x, compute_sam_feat=False):
        encoding_outputs = []
        # 获取MAE的特征
        if compute_sam_feat:
            # 获取 SAM 的特征
            with torch.no_grad():
                sam_feat = self.samEncoder_model(in_x)
        else:
            sam_feat = None
        # 使用UNetHead进行分割
        e1, e2, e3, e4, e5 = self.encoder(x)

        vit_e5 = self.vit(e5)
        vit_e5 = rearrange(vit_e5, "b (x y) c -> b c x y", x=16, y=16)

        # 前景和背景特征映射
        foreground_vector = self.fc_f(vit_e5)
        background_vector = self.fc_b(vit_e5)

        # # 仅在跨层连接部分加 CBAM
        # e1 = self.cbam_e1(e1)
        # e2 = self.cbam_e2(e2)
        # e3 = self.cbam_e3(e3)
        # e4 = self.cbam_e4(e4)
        # 仅在跨层连接部分加 CBAM
        e1 = self.TPAM_e1(e1)
        e2 = self.TPAM_e2(e2)
        e3 = self.TPAM_e3(e3)
        e4 = self.TPAM_e4(e4)

        # # 获取 e1 的空间尺寸
        # target_size = e1.shape[-2:]
        #
        # # 上采样到 e1 尺寸
        # e2_up = F.interpolate(e2, size=target_size, mode='bilinear', align_corners=False)
        # e3_up = F.interpolate(e3, size=target_size, mode='bilinear', align_corners=False)
        # e4_up = F.interpolate(e4, size=target_size, mode='bilinear', align_corners=False)
        #
        # # 拼接通道维度
        # concat_feat = torch.cat([e1, e2_up, e3_up, e4_up], dim=1)  # (8, n, 256, 256)
        #
        # # 计算均值
        # mean_feat = concat_feat.mean(dim=1, keepdim=True)  # (8, 1, 256, 256)

        # encoding_outputs.extend([foreground_vector])
        encoding_outputs.extend([e1, e2, e3, e4, foreground_vector])
        segmentation_output = self.decoder(encoding_outputs)
        # 注册hook来保存梯度
        # if compute_sam_feat:
        #     e1.register_hook(self.save_gradient("e1"))
        #     e2.register_hook(self.save_gradient("e2"))
        #     e3.register_hook(self.save_gradient("e3"))
        #     e4.register_hook(self.save_gradient("e4"))
        #     segmentation_output.register_hook(self.save_gradient("o4"))
            # mean_feat.register_hook(self.save_gradient("m4"))
        return sam_feat, foreground_vector, background_vector, segmentation_output, [e1, e2, e3, e4, segmentation_output]


import torch
from thop import profile, clever_format
from monai.networks.nets import SwinUNETR
from model_sam.build_sam import sam_model_registry
# 假设你已经定义了模型（以 AG_DSV_UNet 为例）
sam = sam_model_registry["vit_b"]().to('cpu')
model = Samunet_Segmentation_Model(sam, 1, 1)

# 将模型放到CPU上（或与输入同一设备）
model.eval()  # 设置为评估模式，避免影响BN等

# 创建一个虚拟输入，尺寸与真实数据一致（batch_size=1, channel=1, height=256, width=256）
dummy_input = torch.randn(1, 3, 256, 256)
dummy_input2 = torch.randn(1, 1, 256, 256)

# 计算FLOPs和参数量
flops, params = profile(model, inputs=(dummy_input,dummy_input2), verbose=False)
flops, params = clever_format([flops, params], "%.3f")  # 格式化为易读形式（例如 32.66G）

print(f"Total FLOPs: {flops}, Total Params: {params}")