#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoModel, AutoConfig
from grid_tagging.gazetteer_fusion import CategoryAttentionFusion, GazetteerFusion
from grid_tagging.GridTaggingProcessor import RelationMetric
from dual_gcn import DualGCN, AlignmentProcessor


class MultiHeadAttention(nn.Module):
    """多头注意力机制（基于 PyTorch SDPA）"""

    def __init__(self, num_heads, hidden_size, attention_probs_dropout_prob):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.attn_dropout_p = attention_probs_dropout_prob


    def _shape(self, x):
        # (B, L, H) -> (B, num_heads, L, head_dim)
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, query, key, value, attention_mask=None):
        # 线性映射
        q = self._shape(self.q_proj(query))
        k = self._shape(self.k_proj(key))
        v = self._shape(self.v_proj(value))

        # 注意：attention_mask 期望为 bool 掩码（True=屏蔽），或加法掩码
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=False,
        )

        # (B, num_heads, L, head_dim) -> (B, L, H)
        context = attn.transpose(1, 2).contiguous().view(query.size(0), query.size(1), self.hidden_size)
        context = self.out_proj(context)

        return context

class HateDetectionGridTagging(nn.Module):
    """的仇恨检测Grid Tagging模型"""

    def __init__(self, config, model_name="hfl/chinese-roberta-wwm-ext"):
        super(HateDetectionGridTagging, self).__init__()

        # 加载BERT模型
        self.bert = AutoModel.from_pretrained(model_name)
        bert_config = AutoConfig.from_pretrained(model_name)

        self.cfg = config

        # 从配置文件读取参数
        self.inner_dim = getattr(config, 'inner_dim', 256)
        self.use_rope = getattr(config, 'use_rope', True)
        self.use_attention = getattr(config, 'use_attention', True)


        # 从配置文件读取标签定义
        self.entity_labels = config.entity_labels
        self.relation_labels = config.relation_labels
        self.polarity_labels = config.polarity_labels
        self.category_labels = config.categories
        # 二分类
        # self.category_labels = config.category_combines

        # 从配置文件读取损失权重（可选）
        self.loss_weight = getattr(config, 'loss_weights', None)
        # 可选：从配置读取正例权重（用于BCE场景）
        self.pos_weights = getattr(config, 'pos_weights', None)

        self.grid_tagging = RelationMetric(config)

        # Gazetteer category-attention fusion (optional)
        self.use_cat_attn_fusion = bool(getattr(config, 'use_cat_attention_fusion', False))
        if self.use_cat_attn_fusion:
            num_categories = len(self.category_labels)
            cat_emb_dim = int(getattr(config, 'cat_attention_emb_dim', 64))
            proj_dim = getattr(config, 'cat_attention_proj_dim', None)
            dropout = float(getattr(config, 'cat_attention_dropout', 0.1))
            self.cat_attention_fusion = CategoryAttentionFusion(
                hidden_size=AutoConfig.from_pretrained(model_name).hidden_size,
                num_categories=num_categories,
                cat_emb_dim=cat_emb_dim,
                proj_dim=proj_dim,
                dropout=dropout,
                config=config,  # 传递配置参数以支持权重配置
            )

        # Gazetteer token-feature fusion (optional)
        self.use_gazetteer_fusion = bool(getattr(config, 'use_gazetteer_fusion', False))
        if self.use_gazetteer_fusion:
            num_categories = len(self.category_labels)
            bio_emb_dim = int(getattr(config, 'gaz_bio_emb_dim', 16))
            cat_proj_dim = int(getattr(config, 'gaz_cat_proj_dim', 32))
            gaz_proj_dim = getattr(config, 'gaz_proj_dim', None)
            gaz_dropout = float(getattr(config, 'gaz_dropout', 0.1))
            self.gazetteer_fusion = GazetteerFusion(
                hidden_size=bert_config.hidden_size,
                num_categories=num_categories,
                bio_emb_dim=bio_emb_dim,
                cat_proj_dim=cat_proj_dim,
                gaz_proj_dim=gaz_proj_dim,
                dropout=gaz_dropout,
            )

        dense_output_dim = self.inner_dim * 4

        self.ent_dim = dense_output_dim * len(self.entity_labels)
        self.rel_dim = dense_output_dim * len(self.relation_labels)
        self.pol_dim = dense_output_dim * len(self.polarity_labels)
        self.cat_dim = dense_output_dim * len(self.category_labels)

        self.dense_all = nn.Linear(bert_config.hidden_size, self.ent_dim + self.rel_dim + self.pol_dim + self.cat_dim)

        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        # 多头注意力机制（如果启用）
        if self.use_attention:
            cfg_heads = getattr(config, 'attention_heads', None)
            num_heads = cfg_heads if cfg_heads is not None else bert_config.num_attention_heads
            self.self_attention = MultiHeadAttention(
                num_heads=num_heads,
                hidden_size=bert_config.hidden_size,
                attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob,
            )
            self.attn_ln = nn.LayerNorm(bert_config.hidden_size)



    def custom_sinusoidal_position_embedding(self, token_index, pos_type):
        """
        自定义正弦位置编码 (RoPE)
        """
        output_dim = self.inner_dim
        assert output_dim % 2 == 0, "inner_dim must be even for RoPE"
        position_ids = token_index.to(dtype=torch.float).unsqueeze(-1)

        half_dim = output_dim // 2
        base = 10000.0 if pos_type == 0 else 15.0
        inv_freq = torch.pow(
            torch.tensor(base, device=token_index.device, dtype=torch.float),
            -2 * torch.arange(0, half_dim, device=token_index.device, dtype=torch.float) / output_dim,
        )

        angles = position_ids * inv_freq  # [len, half_dim]
        pos = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1).reshape(len(token_index), output_dim)
        return pos

    def get_position_embedding_with_rope(self, qw, kw, token_index, pos_type):
        """
        使用RoPE的位置编码 
        """
        batch_size, seq_len, hidden_dim, num_classes = qw.shape

        # 创建位置索引
        if token_index is None:
            token_index = torch.arange(seq_len, device=qw.device).unsqueeze(0).expand(batch_size, -1)

        # 使用新的批量处理方法
        return self.get_ro_embedding(qw, kw, token_index, pos_type)

    def get_instance_embedding(self, qw: torch.Tensor, kw: torch.Tensor, token_index, pos_type):
        """
        基于DiaASQ的RoPE实现
        参数:
        qw : torch.Tensor, (seq_len, num_classes, hidden_size)
        kw : torch.Tensor, (seq_len, num_classes, hidden_size)
        token_index : torch.Tensor, (seq_len,) 位置索引
        pos_type : int, 位置编码类型 (0: token-level, 1: utterance-level)
        """
        seq_len, num_classes = qw.shape[:2]

        # 初始化logits矩阵
        logits = qw.new_zeros([seq_len, seq_len, num_classes])

        # 对每个位置对计算RoPE（移除线程分割逻辑）
        for i in range(seq_len):
            for j in range(seq_len):
                # 简化的相对位置计算（移除线程概念）
                x = token_index[i:i + 1]  # 当前query位置
                y = token_index[j:j + 1]  # 当前key位置

                # RoPE位置编码
                x_pos_emb = self.custom_sinusoidal_position_embedding(x, pos_type)
                y_pos_emb = self.custom_sinusoidal_position_embedding(y, pos_type)

                # RoPE旋转位置编码 (参考DiaASQ实现)
                x_cos_pos = x_pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
                x_sin_pos = x_pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
                cur_qw = qw[i:i + 1]  # [1, num_classes, hidden_size]
                cur_qw2 = torch.stack([-cur_qw[..., 1::2], cur_qw[..., ::2]], -1)
                cur_qw2 = cur_qw2.reshape(cur_qw.shape)
                cur_qw = cur_qw * x_cos_pos + cur_qw2 * x_sin_pos

                y_cos_pos = y_pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
                y_sin_pos = y_pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
                cur_kw = kw[j:j + 1]  # [1, num_classes, hidden_size]
                cur_kw2 = torch.stack([-cur_kw[..., 1::2], cur_kw[..., ::2]], -1)
                cur_kw2 = cur_kw2.reshape(cur_kw.shape)
                cur_kw = cur_kw * y_cos_pos + cur_kw2 * y_sin_pos

                # 计算注意力分数 (参考DiaASQ的einsum模式)
                pred_logits = torch.einsum('mhd,nhd->mnh', cur_qw, cur_kw).contiguous()
                # scale dot-product by 1/sqrt(D) for stability
                D_local = cur_qw.shape[-1]
                scale = cur_qw.new_tensor(float(D_local)).rsqrt()
                pred_logits = pred_logits * scale
                logits[i, j] = pred_logits.squeeze(0).squeeze(0)

        return logits

    def get_ro_embedding(self, qw, kw, token_index, pos_type):
        """
        批量处理RoPE编码
        """
        B, L, H, D = qw.shape
        assert D % 2 == 0, "inner_dim must be even for RoPE"

        if token_index is None:
            token_index = torch.arange(L, device=qw.device).unsqueeze(0).expand(B, -1)
        token_index = token_index.to(dtype=qw.dtype)

        half_dim = D // 2
        base = 10000.0 if pos_type == 0 else 15.0
        inv_freq = torch.pow(
            torch.tensor(base, device=qw.device, dtype=qw.dtype),
            -2 * torch.arange(0, half_dim, device=qw.device, dtype=qw.dtype) / D,
        )  # [half_dim]

        angles = token_index.unsqueeze(-1) * inv_freq  # [B, L, half_dim]
        cos = torch.cos(angles).repeat_interleave(2, dim=-1).unsqueeze(2)  # [B, L, 1, D]
        sin = torch.sin(angles).repeat_interleave(2, dim=-1).unsqueeze(2)  # [B, L, 1, D]

        def rotate_pair(x):
            x_even = x[..., ::2]
            x_odd = x[..., 1::2]
            return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

        q_rot = qw * cos + rotate_pair(qw) * sin
        k_rot = kw * cos + rotate_pair(kw) * sin

        logits = torch.einsum('bmhd,bnhd->bmnh', q_rot, k_rot).contiguous()
        # scale dot-product by 1/sqrt(D) for stability
        scale = qw.new_tensor(float(D)).rsqrt()
        logits = logits * scale
        return logits

    def classify_matrix(self, kwargs, sequence_outputs, mat_name='ent'):
        """
        核心的矩阵分类方法，根据配置选择是否使用RoPE
        """
        # 获取输入标签和掩码
        input_labels = kwargs.get(f"{mat_name}_matrix", None)

        masks = kwargs.get(f"{mat_name}_full_masks", None)

        outputs = torch.split(sequence_outputs, self.inner_dim * 4, dim=-1)
        outputs = torch.stack(outputs, dim=-2)

        q_token, q_utterance, k_token, k_utterance = torch.split(outputs, self.inner_dim, dim=-1)

        if self.use_rope:
            # 计算token级别的相对位置编码
            token_index = kwargs.get('token_index', None)
            pred_logits = self.get_position_embedding_with_rope(q_token, k_token, token_index, pos_type=0)

            # 对于非实体任务，添加utterance级别的编码
            if mat_name != 'ent':
                pred_logits1 = self.get_position_embedding_with_rope(q_utterance, k_utterance, token_index, pos_type=1)
                pred_logits += pred_logits1
        else:
            # 不使用RoPE时，使用dot-product attention
            pred_logits = torch.einsum('bmhd,bnhd->bmnh', q_token, k_token).contiguous()
            # scale dot-product by 1/sqrt(inner_dim) for stability
            scale = q_token.new_tensor(float(self.inner_dim)).rsqrt()
            pred_logits = pred_logits * scale

        nums = pred_logits.shape[-1]

        # pred_logits: [B, L, L, num_labels]
        # input_labels: [B, L, L, num_labels]
        # masks: [B, L, L, num_labels]

        if mat_name == 'cat':
            active_loss = masks[..., 0].view(-1) == 1
            active_logits = pred_logits.view(-1, nums)[active_loss]
            active_labels = input_labels.view(-1, nums)[active_loss].float()
            use_pw = getattr(self.cfg, 'use_pos_weight', None)
            has_pw = self.pos_weights is not None and 'cat' in self.pos_weights
            enable_pw = bool(use_pw.get('cat', False)) if isinstance(use_pw, dict) else False
            if enable_pw and has_pw:
                pos_weight_vec = sequence_outputs.new_tensor(self.pos_weights['cat'])
                criterion = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight_vec)
            else:
                criterion = nn.BCEWithLogitsLoss(reduction="mean")
            loss = criterion(active_logits, active_labels)
        elif mat_name == 'cat_2D':
            input_labels_2D = kwargs.get('categories_matrix_2D', None)
            masks_2D = kwargs.get('cat_full_masks_2D', None)
            active_loss = masks_2D.view(-1) == 1
            criterion = nn.CrossEntropyLoss(
                weight=sequence_outputs.new_tensor(getattr(self.cfg, f"{mat_name}_class_weights")))
            active_logits = pred_logits.view(-1, pred_logits.shape[-1])[active_loss]
            active_labels = input_labels_2D.view(-1)[active_loss]
            loss = criterion(active_logits, active_labels)
        else:
            # 其它头保持不变（rel/ent/pol）
            active_loss = masks[..., 0].view(-1) == 1
            criterion = nn.CrossEntropyLoss(
                weight=sequence_outputs.new_tensor(getattr(self.cfg, f"{mat_name}_class_weights")))
            active_logits = pred_logits.view(-1, pred_logits.shape[-1])[active_loss]
            active_labels = input_labels.argmax(-1).view(-1)[active_loss]
            loss = criterion(active_logits, active_labels)

        return loss, pred_logits

    def build_attention(self, sequence_outputs, attention_mask=None):
        """
        构建自注意力机制
        """
        if not self.use_attention:
            return sequence_outputs

        if attention_mask is not None:
            # 转为 bool 掩码：[B, 1, 1, L]，True 表示屏蔽
            extended_attention_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            attended_outputs = self.self_attention(
                sequence_outputs, sequence_outputs, sequence_outputs, extended_attention_mask
            )
        else:
            attended_outputs = self.self_attention(sequence_outputs, sequence_outputs, sequence_outputs)

        # 残差 + LayerNorm
        outputs = self.attn_ln(sequence_outputs + attended_outputs)
        return outputs

    def forward(self, **kwargs):
        """
        前向传播
        """
        # 获取输入
        global dgcn_outputs, gazetteer_outputs, cat_attn_outputs
        input_ids = kwargs.get('bert_ids')
        token_type_ids = kwargs.get('token_type_ids')
        attention_mask = kwargs.get('input_masks')
        # BERT编码
        sequence_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]
        sequence_outputs = self.dropout(sequence_outputs)

        base = sequence_outputs
        feats = [base]   # 至少包含 BERT 本身

        # 先应用基于 token 的 Gazetteer 融合（若启用）
        if self.use_gazetteer_fusion:
            bio_ids = kwargs.get('bio_ids', None)
            cat_multihot = kwargs.get('cat_multihot', None)
            if bio_ids is not None and cat_multihot is not None:
                gazetteer_outputs = self.gazetteer_fusion(
                    x=sequence_outputs,
                    bio_ids=bio_ids,
                    cat_multihot=cat_multihot,
                )
                feats.append(gazetteer_outputs)

        # 再应用类别注意力融合（若启用）
        if self.use_cat_attn_fusion:
            tc_spans = kwargs.get('tc_spans', None)
            tc_prior = kwargs.get('tc_prior', None)
            bio_ids = kwargs.get('bio_ids', None)
            boundary_mask = (bio_ids == 2).long()
            cat_attn_outputs = self.cat_attention_fusion(
                x=sequence_outputs,
                tc_spans=tc_spans,
                tc_prior=tc_prior,
                boundary_mask=boundary_mask
            )
            feats.append(cat_attn_outputs)

        sequence_outputs = sum(feats)

        # 构建注意力（如果启用）
        # if self.use_attention:
        #     sequence_outputs = self.build_attention(sequence_outputs, attention_mask)
        sequence_outputs = self.dense_all(sequence_outputs)
        sequence_ent = sequence_outputs[:, :, :self.ent_dim]
        sequence_rel = sequence_outputs[:, :, self.ent_dim:self.ent_dim + self.rel_dim]
        sequence_pol = sequence_outputs[:, :, self.ent_dim + self.rel_dim:self.ent_dim + self.rel_dim + self.pol_dim]
        sequence_cat = sequence_outputs[:, :, self.ent_dim + self.rel_dim + self.pol_dim:]


        # 实体分类 (单标签, argmax 解码)
        ent_loss, ent_preds = self.classify_matrix(kwargs, sequence_ent, 'ent')

        # 关系分类 (单标签, argmax 解码)
        rel_loss, rel_preds = self.classify_matrix(kwargs, sequence_rel, 'rel')

        # 极性分类 (单标签, argmax 解码)
        pol_loss, pol_preds = self.classify_matrix(kwargs, sequence_pol, 'pol')

        # 类别分类 (多标签, sigmoid + threshold + fallback 解码)
        cat_loss, cat_preds = self.classify_matrix(kwargs, sequence_cat, 'cat')
        # cat_loss, cat_preds = self.classify_matrix(kwargs, sequence_cat, 'cat_2D')

        self.grid_tagging.add_instance(kwargs, ent_preds, rel_preds, pol_preds, cat_preds)

        return (ent_loss, rel_loss, pol_loss, cat_loss), (ent_preds, rel_preds, pol_preds, cat_preds)


def initialize_model(config, model_name="hfl/chinese-roberta-wwm-ext", device='cpu'):
    """初始化仇恨检测Grid Tagging模型"""
    model = HateDetectionGridTagging(config, model_name)
    return model.to(device)

