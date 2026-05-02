import numpy as np
import pickle
import os
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, ndcg_score, log_loss
import pandas as pd
from scipy.stats import rankdata
import warnings
warnings.filterwarnings("ignore")

'''1：多模态特征自适应融合模块
''' 
class ModalAdaptiveFusion(nn.Module):
    def __init__(self, image_dim=1024, text_dim=1024, video_dim=768, hidden_dim=512, user_emb_dim=64):
        super().__init__()
        # 模态特征统一映射到相同维度 + Dropout
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.Dropout(0.3)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.Dropout(0.2)
        )
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.Dropout(0.2)
        )
        
        # 模态门控：先不用user_emb，改用三个模态特征的平均拼接
        self.modal_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 输入改为模态平均特征
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        # 模态间交叉注意力
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # 输出层 + Dropout
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.2)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, user_emb, image_feat, text_feat, video_feat):
        # 1. 统一模态维度
        image_hid = self.image_proj(image_feat)  # [B, hidden_dim]
        text_hid = self.text_proj(text_feat)
        video_hid = self.video_proj(video_feat)
        
        # 2. 生成模态权重（改用模态特征平均，解决user_emb冷启动）
        modal_avg = (image_hid + text_hid + video_hid) / 3.0
        modal_weights = self.modal_gate(modal_avg)  # [B, 3]
        w_image, w_text, w_video = modal_weights[:, 0:1], modal_weights[:, 1:2], modal_weights[:, 2:3]
        
        # 3. 模态间交叉注意力（建模标题-封面-视频的匹配度）
        modal_seq = torch.stack([image_hid, text_hid, video_hid], dim=1)  # [B, 3, hidden_dim] stack：把多个「同形状的独立张量」变成一个「序列 / 集合
        attn_out, _ = self.cross_attn(modal_seq, modal_seq, modal_seq) # 这里其实就是Q,K,V
        attn_out = attn_out.mean(dim=1)  # [B, hidden_dim] 均值池化得到融合特征 维度为和单模态特征一致
        
        # 4. 自适应加权融合
        fused_feat = w_image * image_hid + w_text * text_hid + w_video * video_hid + attn_out
        
        # 5. 输出归一化
        output = self.layer_norm(self.output_proj(fused_feat))
        return output, modal_weights, image_hid, text_hid, video_hid
    
'''2、长短期兴趣解耦的时序编码模块
'''
class LongShortTermInterestEncoder(nn.Module):
    def __init__(self, item_emb_dim=512, hidden_dim=512, max_seq_len=10, num_heads=4):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        
        # 长期兴趣编码：门控过滤噪声
        self.long_term_gate = nn.Sequential(
            nn.Linear(item_emb_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.long_term_proj = nn.Linear(item_emb_dim, hidden_dim)
        
        # 短期兴趣编码：带因果掩码的Transformer层
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*2, batch_first=True, dropout=0.1
        )
        self.short_term_transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)
        
        # 兴趣融合门控：用目标item自适应加权长短期兴趣
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, item_seq_emb, seq_len, target_item_emb):
        batch_size = item_seq_emb.shape[0]
        
        # ---------------------- 1. 长期兴趣编码 ----------------------
        # 生成padding掩码，过滤无效序列
        mask = torch.arange(self.max_seq_len, device=item_seq_emb.device)[None, :] < seq_len[:, None]
        mask = mask.unsqueeze(-1).float()  # [B, max_seq_len, 1]
        
        # 门控过滤噪声，只保留用户稳定的兴趣
        long_term_gate = self.long_term_gate(item_seq_emb)  # [B, max_seq_len, hidden_dim]
        long_term_feat = self.long_term_proj(item_seq_emb) * long_term_gate * mask
        long_term_feat = long_term_feat.sum(dim=1) / (seq_len.unsqueeze(-1) + 1e-8)  # [B, hidden_dim]
        
        # ---------------------- 2. 短期兴趣编码 ----------------------
        # 加入位置编码，捕捉时序顺序
        pos_ids = torch.arange(self.max_seq_len, device=item_seq_emb.device).unsqueeze(0).repeat(batch_size, 1)
        pos_emb = self.pos_embedding(pos_ids)  # [B, max_seq_len, hidden_dim]
        short_term_input = self.long_term_proj(item_seq_emb) + pos_emb
        
        # 因果掩码：Transformer只能看到当前行为之前的序列，避免未来信息泄露
        causal_mask = nn.Transformer.generate_square_subsequent_mask(self.max_seq_len, device=item_seq_emb.device).bool()
        # padding掩码
        key_padding_mask = ~(torch.arange(self.max_seq_len, device=item_seq_emb.device)[None, :] < seq_len[:, None])
        
        # Transformer编码短期兴趣
        short_term_feat = self.short_term_transformer(
            short_term_input, mask=causal_mask, src_key_padding_mask=key_padding_mask
        )
        # 取最后一个有效位置的输出，作为用户实时短期兴趣
        last_idx = (seq_len - 1).clamp(min=0)
        short_term_feat = short_term_feat[torch.arange(batch_size), last_idx]  # [B, hidden_dim]
        
        # ---------------------- 3. 自适应融合长短期兴趣 ----------------------
        # 用目标item，学习长短期兴趣的融合权重
        fusion_input = torch.cat([long_term_feat, short_term_feat, target_item_emb], dim=-1)
        fusion_weights = self.fusion_gate(fusion_input)  # [B, 2]
        w_long, w_short = fusion_weights[:, 0:1], fusion_weights[:, 1:2]
        
        final_user_interest = w_long * long_term_feat + w_short * short_term_feat
        final_user_interest = self.layer_norm(final_user_interest)
        
        return final_user_interest, long_term_feat, short_term_feat
    
'''3：多模态感知的多目标专家网络（增加Dropout）
'''
class ModalAwareMultiExpert(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, num_targets=1, num_shared_experts=2):
        super().__init__()
        self.num_targets = num_targets
        self.num_shared_experts = num_shared_experts
        
        # 1. 每个目标的专属多模态专家 + Dropout
        self.target_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(0.3),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_targets)
        ])
        
        # 2. 共享专家，建模目标共性 + Dropout
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(0.3),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_shared_experts)
        ])
        
        # 3. 每个目标的门控网络，自适应选择专家
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_targets + num_shared_experts),
                nn.Softmax(dim=-1)
            ) for _ in range(num_targets)
        ])
        
        # 4. 每个目标的预测头
        self.predict_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_targets)
        ])
        
    def forward(self, fusion_feat):
        # 1. 所有专家的输出
        expert_outputs = []
        # 目标专属专家输出
        for expert in self.target_experts:
            expert_outputs.append(expert(fusion_feat).unsqueeze(1))  # [B, 1, hidden_dim]
        # 共享专家输出
        for expert in self.shared_experts:
            expert_outputs.append(expert(fusion_feat).unsqueeze(1))
        
        expert_outputs = torch.cat(expert_outputs, dim=1)  # [B, num_experts, hidden_dim]
        
        # 2. 每个目标的门控加权
        final_outputs = []
        for i in range(self.num_targets):
            gate_weights = self.gates[i](fusion_feat).unsqueeze(-1)  # [B, num_experts, 1]
            weighted_feat = (expert_outputs * gate_weights).sum(dim=1)  # [B, hidden_dim]
            pred = self.predict_heads[i](weighted_feat)
            final_outputs.append(pred)
        
        return torch.cat(final_outputs, dim=-1)  # [B, num_targets]
    
'''4：轻量级对比学习辅助正则模块（暂时不用）
'''
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.2):  # 固定温度系数0.2，避免数值溢出
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, feat1, feat2):
        batch_size = feat1.shape[0]
        # 防止batch_size=1时损失错误
        if batch_size <= 1:
            return torch.tensor(0.0, device=feat1.device)
        
        # 特征归一化，加小epsilon防止除0
        feat1 = nn.functional.normalize(feat1, dim=-1, eps=1e-8)
        feat2 = nn.functional.normalize(feat2, dim=-1, eps=1e-8)
        
        # 计算相似度矩阵，减去最大值稳定softmax，防止inf
        logits = torch.matmul(feat1, feat2.T) / self.temperature
        logits = logits - logits.max(dim=-1, keepdim=True)[0].detach()
        
        # 正例标签
        labels = torch.arange(batch_size, device=feat1.device)
        
        # 双向对比损失
        loss = (self.cross_entropy(logits, labels) + self.cross_entropy(logits.T, labels)) / 2
        return loss  

'''5 多模态多目标推荐模型（增加Embedding Dropout + 全局pos_weight）
'''
class MMISD_RecommendModel(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        image_embedding,
        text_embedding,
        video_embedding,
        pad_id,
        global_pos_weight,  # 新增全局pos_weight
        user_emb_dim=64,
        item_emb_dim=512,
        hidden_dim=512,
        image_dim=1024,
        text_dim=1024,
        video_dim=768,
        max_seq_len=10,  
        num_targets=1,
        num_shared_experts=2,
        temperature=0.2
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_targets = num_targets
        self.global_pos_weight = global_pos_weight  # 保存全局pos_weight
        
        # 1. 基础Embedding层 + Dropout
        self.user_embedding = nn.Sequential(
            nn.Embedding(n_users, user_emb_dim),
            nn.Dropout(0.2)
        )
        self.item_id_embedding = nn.Sequential(
            nn.Embedding(n_items + 1, item_emb_dim, padding_idx=pad_id),
            nn.Dropout(0.15)
        )
        
        # 2. 多模态特征Embedding（冻结预训练特征）
        self.image_embedding = image_embedding
        self.text_embedding = text_embedding
        self.video_embedding = video_embedding
        
        # 3. 核心模块
        self.modal_fusion = ModalAdaptiveFusion(image_dim, text_dim, video_dim, hidden_dim, user_emb_dim)
        self.interest_encoder = LongShortTermInterestEncoder(hidden_dim, hidden_dim, max_seq_len)
        self.multi_expert_network = ModalAwareMultiExpert(hidden_dim*2, hidden_dim, num_targets, num_shared_experts)
        # 4. 对比学习损失（暂时不用）
        self.contrastive_loss = ContrastiveLoss(temperature)
        
        # 5. 统计特征投影 + Dropout
        self.stat_feat_proj = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, hidden_dim),
            nn.Dropout(0.2)
        )
        
    def forward(self, batch, training=True):
        # ---------------------- 1. 基础特征提取 ----------------------
        user_id = batch["user_id"]
        item_id = batch["item_id"]
        history_seq = batch["history_seq"]  # [B, max_seq_len]
        seq_len = batch["seq_len"]  # [B]
        # 视频统计特征（已归一化）
        stat_feat = torch.stack([batch["likes_count"], batch["views_count"], batch["interaction_rate"]], dim=-1)
        stat_feat = self.stat_feat_proj(stat_feat)  # [B, hidden_dim]
        
        # 用户和item ID embedding（带Dropout）
        user_emb = self.user_embedding(user_id)  # [B, user_emb_dim]
        item_id_emb = self.item_id_embedding(item_id)  # [B, item_emb_dim]
        
        # 多模态特征提取
        image_feat = self.image_embedding(item_id)
        text_feat = self.text_embedding(item_id)
        video_feat = self.video_embedding(item_id)
        
        # ---------------------- 2. 多模态自适应融合 ----------------------
        item_fused_feat, modal_weights, image_hid, text_hid, video_hid = self.modal_fusion(user_emb, image_feat, text_feat, video_feat)
        item_final_feat = item_fused_feat + item_id_emb + stat_feat  # 融合ID、统计、多模态特征
        
        # ---------------------- 3. 用户长短期兴趣编码 ----------------------
        # 历史序列的item融合特征
        batch_size, seq_len_max = history_seq.shape
        history_image_feat = self.image_embedding(history_seq.reshape(-1)).reshape(batch_size, seq_len_max, -1)
        history_text_feat = self.text_embedding(history_seq.reshape(-1)).reshape(batch_size, seq_len_max, -1)
        history_video_feat = self.video_embedding(history_seq.reshape(-1)).reshape(batch_size, seq_len_max, -1)
        
        # 历史序列独立投影+平均融合
        history_image_hid = self.modal_fusion.image_proj[0](history_image_feat)  # 取投影层（跳过dropout）
        history_text_hid = self.modal_fusion.text_proj[0](history_text_feat)
        history_video_hid = self.modal_fusion.video_proj[0](history_video_feat)
        history_fused_feat = (history_image_hid + history_text_hid + history_video_hid) / 3.0
        # 加上item id embedding
        history_item_emb = self.item_id_embedding[0](history_seq)  # 取embedding层（跳过dropout）
        history_final_feat = history_fused_feat + history_item_emb
        
        # 编码用户兴趣
        user_interest_feat, long_term, short_term = self.interest_encoder(history_final_feat, seq_len, item_final_feat)
        
        # ---------------------- 4. 多目标预测 ----------------------
        fusion_input = torch.cat([user_interest_feat, item_final_feat], dim=-1)
        pred_logits = self.multi_expert_network(fusion_input)  # [B, num_targets]
        
        # ---------------------- 5. 训练模式：计算损失 ----------------------
        if training:
            # 主损失：多目标二分类交叉熵损失（用全局pos_weight）
            target_labels = batch["targets"]  # [B, num_targets]
            
            # 原有主损失计算
            main_loss = 0
            for i in range(self.num_targets):
                main_loss += nn.functional.binary_cross_entropy_with_logits(
                    pred_logits[:, i], target_labels[:, i], pos_weight=self.global_pos_weight
                )
            main_loss = main_loss / self.num_targets
            
            # 新增：低权重对比学习损失，不影响主任务收敛，提升泛化能力
            modal_contrast_loss = (
                self.contrastive_loss(image_hid, text_hid) + 
                self.contrastive_loss(text_hid, video_hid)
            ) / 2
            user_contrast_loss = self.contrastive_loss(long_term, short_term)
            
            # 总损失：辅助损失权重控制在万分级，避免主导主任务
            total_loss = main_loss + 0.001 * modal_contrast_loss + 0.0005 * user_contrast_loss
            return pred_logits, total_loss
        
        # 推理模式：只返回预测结果
        return pred_logits

'''6. ShortVideoDataset 
'''
class ShortVideoDataset(Dataset):
    def __init__(self, df, max_seq_len=10, pad_id=0):  
        self.df = df
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        # 提前把所有列转成numpy数组，提速
        self.user_ids = df["user_id_enc"].values.astype(np.int64)
        self.item_ids = df["item_id_enc"].values.astype(np.int64)
        self.history_seqs = df["history_seq"].values
        self.likes_counts = df["likes_count"].values.astype(np.float32)
        self.views_counts = df["views_count"].values.astype(np.float32)
        self.interaction_rates = df["interaction_rate"].values.astype(np.float32)
        self.targets = df[["ctr_label"]].values.astype(np.float32)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 直接从内存数组取数
        user_id = self.user_ids[idx]
        item_id = self.item_ids[idx]
        history_seq = self.history_seqs[idx]
        
        # 历史序列处理
        history_seq = np.array(history_seq, dtype=np.int64)
        seq_len = len(history_seq)
        # 截断过长序列（max_seq_len=10）
        if seq_len > self.max_seq_len:
            history_seq = history_seq[-self.max_seq_len:]
        # 用pad_id补全
        else:
            history_seq = np.pad(history_seq, (0, self.max_seq_len - seq_len), mode='constant', constant_values=self.pad_id)
        seq_len = int(min(seq_len, self.max_seq_len))  # 保证是int类型
        
        return {
            "user_id": user_id,
            "item_id": item_id,
            "history_seq": history_seq,
            "seq_len": seq_len,
            "likes_count": self.likes_counts[idx],
            "views_count": self.views_counts[idx],
            "interaction_rate": self.interaction_rates[idx],
            "targets": self.targets[idx]
        }

# ====================== 主执行逻辑 ======================
if __name__ == '__main__':
    # 【Windows多进程必加】显式设置启动方法
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    # 全局CUDA优化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    
    # 你的文件路径
    dir = r'2 项目\2 短视频生成式推荐 4.25-4.2\data\MicroLens-100k'
    processed_dir = os.path.join(dir, "processed_final")
    
    # 1. 加载编码器与辅助数据
    with open(os.path.join(processed_dir, "aux_data.pkl"), "rb") as f:
        aux_data = pickle.load(f)
    user_encoder = aux_data["user_encoder"]
    item_encoder = aux_data["item_encoder"]
    n_items = len(item_encoder.classes_)  # 视频总数19738
    pad_id = n_items  # padding id用n_items
    n_users = len(user_encoder.classes_)
    
    # 2. 加载多模态特征npy文件 + L2归一化
    print("正在加载并归一化多模态特征...")
    image_feat = np.load(os.path.join(dir, "MicroLens-100k_image_features_CLIPRN50.npy"))
    text_feat = np.load(os.path.join(dir, "MicroLens-100k_title_en_text_features_BgeM3.npy"))
    video_feat = np.load(os.path.join(dir, "MicroLens-100k_video_features_VideoMAE.npy"))
    
    # 多模态特征L2归一化
    image_feat = image_feat / (np.linalg.norm(image_feat, axis=-1, keepdims=True) + 1e-8)
    text_feat = text_feat / (np.linalg.norm(text_feat, axis=-1, keepdims=True) + 1e-8)
    video_feat = video_feat / (np.linalg.norm(video_feat, axis=-1, keepdims=True) + 1e-8)
    
    # 3. 构建Embedding层查找表，增加pad_id的0向量
    image_feat_matrix = np.concatenate([image_feat, np.zeros((1, 1024), dtype=np.float32)], axis=0)
    text_feat_matrix = np.concatenate([text_feat, np.zeros((1, 1024), dtype=np.float32)], axis=0)
    video_feat_matrix = np.concatenate([video_feat, np.zeros((1, 768), dtype=np.float32)], axis=0)
    
    # 填充item特征映射
    item_modal_feat_map = {}
    for raw_item_id in item_encoder.classes_:
        enc_id = item_encoder.transform([raw_item_id])[0]
        npy_idx = raw_item_id - 1
        item_modal_feat_map[enc_id] = {
            "image": image_feat[npy_idx],
            "text": text_feat[npy_idx],
            "video": video_feat[npy_idx]
        }
    for enc_id, feats in item_modal_feat_map.items():
        image_feat_matrix[enc_id] = feats["image"]
        text_feat_matrix[enc_id] = feats["text"]
        video_feat_matrix[enc_id] = feats["video"]
    
    # 构建冻结的Embedding层
    image_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(image_feat_matrix), freeze=True, padding_idx=pad_id)
    text_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(text_feat_matrix), freeze=True, padding_idx=pad_id)
    video_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(video_feat_matrix), freeze=True, padding_idx=pad_id)
    print(" 多模态特征映射完成！")
    print(f"图像特征维度：1024，文本特征维度：1024，视频特征维度：768")
    
    # 4. 加载数据集
    print("正在加载数据集...")
    train_df = pd.read_parquet(os.path.join(processed_dir, "train_final.parquet"))
    val_df = pd.read_parquet(os.path.join(processed_dir, "val_final.parquet"))
    test_df = pd.read_parquet(os.path.join(processed_dir, "test_final.parquet"))
    
    # 5. 数据优化：过滤无效样本 + 负样本下采样
    print("正在过滤无效样本...")
    # 过滤历史序列长度<3的样本
    def filter_short_seq(df):
        df["seq_len"] = df["history_seq"].apply(len)
        df = df[df["seq_len"] >= 3].reset_index(drop=True)
        return df
    train_df = filter_short_seq(train_df)
    val_df = filter_short_seq(val_df)
    test_df = filter_short_seq(test_df)
    
    # 6. 构建单目标CTR标签
    def build_multi_target(df):
        df["ctr_label"] = df["label"]
        df["targets"] = df[["ctr_label"]].values.astype(np.float32)
        return df
    train_df = build_multi_target(train_df)
    val_df = build_multi_target(val_df)
    test_df = build_multi_target(test_df)
    
    # 7. 统计特征全局归一化
    print("正在归一化统计特征...")
    stats_cols = ["likes_count", "views_count", "interaction_rate"]
    # 只用训练集统计量，避免数据泄露
    train_mean = train_df[stats_cols].mean()
    train_std = train_df[stats_cols].std() + 1e-8
    # 归一化
    train_df[stats_cols] = (train_df[stats_cols] - train_mean) / train_std
    val_df[stats_cols] = (val_df[stats_cols] - train_mean) / train_std
    test_df[stats_cols] = (test_df[stats_cols] - train_mean) / train_std
    print(" 数据预处理完成！")
    
    # 8. 计算全局pos_weight
    total_pos = train_df["ctr_label"].sum()
    total_neg = len(train_df) - total_pos
    global_pos_weight = torch.tensor(total_neg / (total_pos + 1e-8), device=device)
    print(f"全局正负样本比例：正样本={total_pos:.0f}，负样本={total_neg:.0f}，pos_weight={global_pos_weight.item():.2f}")
    
    # 9. 构建Dataset与DataLoader
    max_seq_len = 10  
    batch_size = 1024  
    num_workers = 2  
    
    train_dataset = ShortVideoDataset(train_df, max_seq_len, pad_id=pad_id)
    val_dataset = ShortVideoDataset(val_df, max_seq_len, pad_id=pad_id)
    test_dataset = ShortVideoDataset(test_df, max_seq_len, pad_id=pad_id)
    
    # 训练集DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )
    # 验证/测试集：batch_size翻倍
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size*2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,  # 进程常驻，避免每个epoch重复创建开销
        prefetch_factor=2  # 预加载2个batch，GPU无等待
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size*2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    print(" DataLoader构建完成！")
    
    # 10. 模型初始化
    model = MMISD_RecommendModel(
        n_users=n_users,
        n_items=n_items,
        image_embedding=image_embedding,
        text_embedding=text_embedding,
        video_embedding=video_embedding,
        pad_id=pad_id,
        global_pos_weight=global_pos_weight,  # 传入全局pos_weight
        max_seq_len=max_seq_len,
        num_targets=1,
        temperature=0.2
    ).to(device)
    
    # 11. 优化器与训练配置
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-5)
    # 改用ReduceLROnPlateau调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.2, patience=1, min_lr=1e-6    
    )
    scaler = torch.amp.GradScaler("cuda")  # 混合精度缩放器
    
    num_epochs = 100
    best_auc = 0
    early_stop_patience = 4  # 早停耐心值
    early_stop_count = 0
    target_names = ["CTR"]
    
    # 12. 训练循环
    print("====================== 开始训练 ======================")
    for epoch in range(num_epochs):
        # ---------------------- 训练阶段 ----------------------
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} 训练", mininterval=1)
        
        for batch in train_pbar:
            # 数据移到GPU
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            optimizer.zero_grad(set_to_none=True)
            # 混合精度前向传播
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pred_logits, loss = model(batch, training=True)
            
            # 跳过loss为nan的batch
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            # 反向传播+梯度裁剪
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # ---------------------- 验证阶段 ----------------------
        model.eval()
        # 预分配GPU张量，避免循环内频繁内存申请，全量在GPU拼接
        all_val_preds = torch.tensor([], device=device, dtype=torch.float16)
        all_val_labels = torch.tensor([], device=device, dtype=torch.float32)
        all_val_user_ids = torch.tensor([], device=device, dtype=torch.int64)

        # 全程无进度条，减少控制台IO开销
        with torch.inference_mode():
            for batch in val_loader:
                # 数据异步搬运到GPU
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                pred_logits = model(batch, training=False)
                
                # GPU上批量拼接，仅最后一次搬到CPU
                all_val_preds = torch.cat([all_val_preds, pred_logits[:, 0]], dim=0)
                all_val_labels = torch.cat([all_val_labels, batch["targets"][:, 0]], dim=0)
                all_val_user_ids = torch.cat([all_val_user_ids, batch["user_id"]], dim=0)

        # 一次性CPU搬运，转为numpy数组
        all_val_preds = all_val_preds.cpu().numpy()
        all_val_labels = all_val_labels.cpu().numpy()
        all_val_user_ids = all_val_user_ids.cpu().numpy()

        # 1. 必算核心指标：AUC
        val_auc = roc_auc_score(all_val_labels, all_val_preds)

        val_ndcg = {5: 0.0, 10: 0.0}
        val_hit_rate = {5: 0.0, 10: 0.0}
        val_precision = {5: 0.0, 10: 0.0}
        val_recall = {5: 0.0, 10: 0.0}

        if val_auc > best_auc:
            K_list = [5, 10]
            # 预计算log2表，避免循环内重复计算，提速30%
            log2_table = np.log2(np.arange(2, 22))  # 覆盖K=5/10的最大长度
            
            # 纯numpy极速分组，比pandas快10倍
            sort_idx = np.argsort(all_val_user_ids)
            sorted_uids = all_val_user_ids[sort_idx]
            sorted_preds = all_val_preds[sort_idx]
            sorted_labels = all_val_labels[sort_idx]
            
            # 找到用户分界点，一次性拆分所有用户
            uid_change = np.where(sorted_uids[:-1] != sorted_uids[1:])[0] + 1
            user_splits = np.split(np.arange(len(sorted_uids)), uid_change)
            
            # 预分配内存，避免append开销
            ndcg_res = {5: [], 10: []}
            hit_res = {5: [], 10: []}
            prec_res = {5: [], 10: []}
            rec_res = {5: [], 10: []}
            
            # 无冗余纯循环计算
            for split in user_splits:
                p = sorted_preds[split]
                lbl = sorted_labels[split]
                pos_sum = lbl.sum()
                
                # 跳过无正样本的无效用户
                if pos_sum == 0:
                    continue
                
                # 一次性排序，所有K共用
                sorted_idx = np.argsort(p)[::-1]
                sorted_lbl = lbl[sorted_idx]
                
                for K in K_list:
                    top_k = min(K, len(sorted_lbl))
                    top_lbl = sorted_lbl[:top_k]
                    top_pos_sum = top_lbl.sum()
                    
                    # 全用预计算表，无重复运算
                    dcg = (top_lbl / log2_table[:top_k]).sum()
                    idcg = (np.sort(lbl)[::-1][:K] / log2_table[:K]).sum()
                    
                    ndcg_res[K].append(dcg / idcg if idcg > 0 else 0.0)
                    hit_res[K].append(1.0 if top_pos_sum > 0 else 0.0)
                    prec_res[K].append(top_pos_sum / K)
                    rec_res[K].append(top_pos_sum / pos_sum)
            
            # 计算最终均值
            for K in K_list:
                val_ndcg[K] = np.mean(ndcg_res[K]) if ndcg_res[K] else 0.0
                val_hit_rate[K] = np.mean(hit_res[K]) if hit_res[K] else 0.0
                val_precision[K] = np.mean(prec_res[K]) if prec_res[K] else 0.0
                val_recall[K] = np.mean(rec_res[K]) if rec_res[K] else 0.0

        # 打印结果
        print(f"\nEpoch {epoch+1} 结果：")
        print(f"训练平均损失：{avg_train_loss:.4f}")
        print(f"验证集 CTR AUC：{val_auc:.4f}")
        print(f"验证集 NDCG@5： {val_ndcg[5]:.4f} | Hit Rate@5： {val_hit_rate[5]:.4f} | Precision@5： {val_precision[5]:.4f} | Recall@5： {val_recall[5]:.4f}")
        print(f"验证集 NDCG@10：{val_ndcg[10]:.4f} | Hit Rate@10：{val_hit_rate[10]:.4f} | Precision@10：{val_precision[10]:.4f} | Recall@10：{val_recall[10]:.4f}")

        # 学习率调度与早停逻辑
        scheduler.step(val_auc)
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_mmisd_model.pth")
            print(f"最优模型已保存，当前最优AUC：{best_auc:.4f}")
            early_stop_count = 0
        else:
            early_stop_count += 1
            print(f"早停计数器：{early_stop_count}/{early_stop_patience}")
            if early_stop_count >= early_stop_patience:
                print(f"连续{early_stop_patience}个epoch AUC未提升，触发早停")
                break
        
    # ---------------------- 测试集最终评估 ----------------------
    print("\n====================== 测试集最终评估 ======================")
    model.load_state_dict(torch.load("best_mmisd_model.pth"))
    model.eval()
    all_test_preds = []
    all_test_labels = []
    all_test_user_ids = []

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="测试集评估", mininterval=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            pred_logits = model(batch, training=False)
            targets = batch["targets"]
            
            all_test_preds.append(pred_logits[:, 0].cpu().numpy())
            all_test_labels.append(targets[:, 0].cpu().numpy())
            all_test_user_ids.append(batch["user_id"].cpu().numpy())

    # 拼接所有结果（一次性拼接，减少内存操作）
    all_test_preds = np.concatenate(all_test_preds)
    all_test_labels = np.concatenate(all_test_labels)
    all_test_user_ids = np.concatenate(all_test_user_ids)

    # 1. 基础指标计算
    test_auc = roc_auc_score(all_test_labels, all_test_preds)

    # 2. Top-K指标优化：pandas分组+向量化计算（替代循环）
    K_list = [5, 10]
    test_ndcg = {}
    test_hit_rate = {}
    test_precision = {}
    test_recall = {}

    # 构建DataFrame，一次性分组
    df_test = pd.DataFrame({
        'user_id': all_test_user_ids,
        'pred': all_test_preds,
        'label': all_test_labels
    })

    # 过滤无正样本的用户组（提前过滤，减少计算量）
    user_pos_count = df_test.groupby('user_id')['label'].sum()
    valid_users = user_pos_count[user_pos_count > 0].index
    df_test_valid = df_test[df_test['user_id'].isin(valid_users)].reset_index(drop=True)

    # 按用户分组批量计算
    grouped = df_test_valid.groupby('user_id')

    # 预定义存储数组
    ndcg_dict = {k: [] for k in K_list}
    hit_dict = {k: [] for k in K_list}
    precision_dict = {k: [] for k in K_list}
    recall_dict = {k: [] for k in K_list}

    # 批量处理每个用户组（向量化计算）
    def calc_test_user_metrics(group):
        preds = group['pred'].values
        labels = group['label'].values
        pos_total = labels.sum()
        
        # 一次性计算所有K的Top-K索引（降序排序）
        sorted_idx = np.argsort(preds)[::-1]
        
        for K in K_list:
            # 取Top-K（处理样本数少于K的情况）
            top_k = min(K, len(sorted_idx))
            top_k_idx = sorted_idx[:top_k]
            top_k_labels = labels[top_k_idx]
            
            # NDCG@K（向量化计算）
            dcg = (top_k_labels / np.log2(np.arange(2, top_k + 2))).sum()
            idcg = (np.sort(labels)[::-1][:K] / np.log2(np.arange(2, K + 2))).sum()
            ndcg = dcg / idcg if idcg > 0 else 0.0
            
            # Hit Rate@K
            hit = 1 if top_k_labels.sum() > 0 else 0.0
            
            # Precision@K
            precision = top_k_labels.sum() / K
            
            # Recall@K
            recall = top_k_labels.sum() / pos_total
            
            ndcg_dict[K].append(ndcg)
            hit_dict[K].append(hit)
            precision_dict[K].append(precision)
            recall_dict[K].append(recall)

    # 批量计算所有用户
    _ = grouped.apply(calc_test_user_metrics)

    # 计算最终均值
    for K in K_list:
        test_ndcg[K] = np.mean(ndcg_dict[K]) if ndcg_dict[K] else 0.0
        test_hit_rate[K] = np.mean(hit_dict[K]) if hit_dict[K] else 0.0
        test_precision[K] = np.mean(precision_dict[K]) if precision_dict[K] else 0.0
        test_recall[K] = np.mean(recall_dict[K]) if recall_dict[K] else 0.0

    # 打印所有指标
    print(f"测试集 AUC：{test_auc:.4f}")
    for K in K_list:
        if K == 5:
            print(f"测试集 NDCG@{K}：{test_ndcg[K]:.4f} | Hit Rate@{K}：{test_hit_rate[K]:.4f} | Precision@{K}：{test_precision[K]:.4f} | Recall@{K}：{test_recall[K]:.4f}")
            continue
        print(f"测试集 NDCG@{K}：{test_ndcg[K]:.4f} | Hit Rate@{K}：{test_hit_rate[K]:.4f} | Precision@{K}：{test_precision[K]:.4f} | Recall@{K}：{test_recall[K]:.4f}")
    print(f"训练完成！最优验证AUC：{best_auc:.4f}，测试集AUC：{test_auc:.4f}")