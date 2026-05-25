# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.modeling.utils import cat

from .utils_motifs import (
    encode_box_info,
    nms_overlaps,
    obj_edge_vectors,
    rel_vectors,
    to_onehot,
)


class SGGLayer(nn.Module):
    """One block of AlignG's prototype-feedback module.

    Updates per-image predicate prototypes from edge (relation) features and
    then recalibrates edge features with the adapted prototypes.
    """

    def __init__(self, num_rel_cls: int, hidden_dim: int, num_heads: int, dropout: float = 0.2):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.R, self.D, self.H = num_rel_cls, hidden_dim, num_heads
        self.d = hidden_dim // num_heads
        self.sqrt_d = math.sqrt(self.d)

        self.dropout = nn.Dropout(dropout)

        # Proto update
        self.proto_norm = nn.LayerNorm(self.D)
        self.proto_q = nn.Linear(self.D, self.D)
        self.edge_k = nn.Linear(self.D, self.D)
        self.edge_v = nn.Linear(self.D, self.D)
        self.proto_gru = nn.GRUCell(self.D, self.D)
        self.update_norm_p = nn.LayerNorm(self.D)

        # Edge update
        self.edge_norm = nn.LayerNorm(self.D)
        self.edge_q = nn.Linear(self.D, self.D)
        self.proto_k = nn.Linear(self.D, self.D)
        self.proto_v = nn.Linear(self.D, self.D)
        self.edge_ffn = nn.Linear(self.D * 2, self.D)

    def forward(self, protos, edge_feat, rel_pair_idxs, offset=0):
        edge_outputs = []
        for pair_idx in rel_pair_idxs:
            num_pairs = pair_idx.size(0)
            edge_feat_img = edge_feat[offset: offset + num_pairs]

            # --- Proto update ---
            protos_img_norm = self.proto_norm(protos)
            q_s = self.proto_q(protos_img_norm).view(self.R, self.H, self.d)
            k_e = self.edge_k(edge_feat_img).view(num_pairs, self.H, self.d)
            v_e = self.edge_v(edge_feat_img).view(num_pairs, self.H, self.d)

            q_s_b = q_s.permute(1, 0, 2)  # [H, R, d]
            k_e_b = k_e.permute(1, 0, 2)  # [H, P, d]
            v_e_b = v_e.permute(1, 0, 2)  # [H, P, d]

            attn_logits = torch.bmm(q_s_b, k_e_b.transpose(1, 2)) / self.sqrt_d
            attn = F.softmax(attn_logits, dim=2)
            attn = self.dropout(attn)

            updates = torch.bmm(attn, v_e_b)  # [H, R, d]
            updates = updates.permute(1, 0, 2).contiguous().view(self.R, self.D)
            updates = self.update_norm_p(updates)

            protos_img = self.proto_gru(updates, protos_img_norm)

            # --- Edge update ---
            edges_img_norm = self.edge_norm(edge_feat_img)
            q_e = self.edge_q(edges_img_norm).view(num_pairs, self.H, self.d)
            k_s = self.proto_k(protos_img).view(self.R, self.H, self.d)
            v_s = self.proto_v(protos_img).view(self.R, self.H, self.d)

            q_e_b = q_e.permute(1, 0, 2)  # [H, P, d]
            k_s_b = k_s.permute(1, 0, 2)  # [H, R, d]
            v_s_b = v_s.permute(1, 0, 2)  # [H, R, d]

            attn2_logits = torch.bmm(q_e_b, k_s_b.transpose(1, 2)) / self.sqrt_d
            attn2 = F.softmax(attn2_logits, dim=2)
            attn2 = self.dropout(attn2)

            e_updates = torch.bmm(attn2, v_s_b)  # [H, P, d]
            e_updates = e_updates.permute(1, 0, 2).contiguous().view(num_pairs, self.D)
            edges_img = self.edge_ffn(torch.cat([edges_img_norm, e_updates], dim=-1))

            edge_outputs.append(edges_img)
            offset += num_pairs

        return torch.cat(edge_outputs, dim=0)


@registry.ROI_RELATION_PREDICTOR.register("SGGPredictor")
class SGGHead(nn.Module):
    """AlignG predictor: context-conditioned predicate prototypes via prototype feedback."""

    def __init__(self, cfg, in_channels: int):
        super().__init__()
        self.cfg = cfg
        self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        self.dropout_p = 0.2
        self.embed_dim = 300
        self.mlp_dim = 2048
        self.pool_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.post_emb = nn.Linear(in_channels, self.mlp_dim * 2)

        use_gt = cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX
        use_gt_lbl = cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        self.mode = 'predcls' if use_gt and use_gt_lbl else 'sgcls' if use_gt else 'sgdet'

        # GloVe embeddings for objects and predicates
        stats = get_dataset_statistics(cfg)
        obj_vecs = obj_edge_vectors(stats['obj_classes'], wv_dir=cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        rel_vecs = rel_vectors(stats['rel_classes'], wv_dir=cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_vecs)
            self.rel_embed.weight.copy_(rel_vecs)

        # Visual-semantic gates
        self.gate_sub = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_obj = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_union = nn.Linear(self.mlp_dim * 2, self.mlp_dim)

        self.vis2sem = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim * 2), nn.ReLU(True),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.mlp_dim * 2, self.mlp_dim),
        )

        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)

        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)

        self.dropout_sub = nn.Dropout(self.dropout_p)
        self.dropout_obj = nn.Dropout(self.dropout_p)
        self.dropout_rel_rep = nn.Dropout(self.dropout_p)

        self.dropout_rel = nn.Dropout(self.dropout_p)
        self.dropout_pred = nn.Dropout(self.dropout_p)

        self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim * 2, 2)

        # Positional encoding for entity boxes
        self.pos_embed = nn.Sequential(
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        )
        self.down_samp = MLP(self.pool_dim, self.mlp_dim, self.mlp_dim, 2)

        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        heads = 2
        self.SGGLayer = SGGLayer(
            self.num_rel_cls,
            self.mlp_dim,
            heads,
            dropout=self.dropout_p,
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.obj_dim = in_channels
        self.out_obj = make_fc(self.mlp_dim, self.num_obj_cls)
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.mlp_dim)
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys,
                roi_features, union_features, logger=None):
        add_losses = {}
        ent_dists, ent_preds = self.refine_obj_labels(roi_features, proposals)

        entity_rep = self.post_emb(roi_features)
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)

        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)

        ent_embeds = self.obj_embed(ent_preds)

        num_objs = [len(p) for p in proposals]
        num_rels = [r.size(0) for r in rel_pair_idxs]

        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        node_embeds = ent_embeds.split(num_objs, 0)
        ent_preds = ent_preds.split(num_objs, dim=0)

        rel_rep = []
        for pair_idx, sub_rep, obj_rep, entity_embed in zip(rel_pair_idxs, sub_reps, obj_reps, node_embeds):
            s_embed = self.W_sub(entity_embed[pair_idx[:, 0]])  # Ws · ts
            o_embed = self.W_obj(entity_embed[pair_idx[:, 1]])  # Wo · to

            sem_sub = self.vis2sem(sub_rep[pair_idx[:, 0]])  # h(xs)
            sem_obj = self.vis2sem(obj_rep[pair_idx[:, 1]])  # h(xo)

            gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))
            gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))

            sub = s_embed + sem_sub * gate_sem_sub  # s = Ws·ts + gs · h(xs)
            obj = o_embed + sem_obj * gate_sem_obj  # o = Wo·to + go · h(xo)

            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)

            rel_rep.append(fusion_func(sub, obj))
        rel_rep = torch.cat(rel_rep, 0)

        sem_pred = self.vis2sem(self.down_samp(union_features))  # h(xu)
        gate_sem_pred = torch.sigmoid(self.gate_pred(cat((rel_rep, sem_pred), dim=-1)))
        rel_rep = rel_rep - sem_pred * gate_sem_pred  # r = F(s,o) - gp · h(xu)
        rel_rep = self.norm_rel_rep(rel_rep + self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))))

        predicate_proto = self.W_pred(self.rel_embed.weight)
        rel_rep = self.SGGLayer(predicate_proto, rel_rep, rel_pair_idxs)

        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))

        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)

        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()

        if self.training:
            rel_labels = torch.cat(rel_labels)

            # Prototype regularization: cosine similarity
            target_proto_norm = predicate_proto_norm.clone().detach()
            simil_mat = predicate_proto_norm @ target_proto_norm.t()
            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (self.num_rel_cls * self.num_rel_cls)
            add_losses["l21_loss"] = l21 * self.cfg.MODEL.ROI_RELATION_HEAD.LOSS_WEIGHT_L21

            # Prototype regularization: Euclidean distance (push prototypes apart)
            gamma2 = 20.0
            predicate_proto_a = predicate_proto.unsqueeze(1).expand(-1, self.num_rel_cls, -1)
            predicate_proto_b = predicate_proto.detach().unsqueeze(0).expand(self.num_rel_cls, -1, -1)
            proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2
            sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
            # Self-distance sits at index 0 after sorting; index 1 is the nearest other prototype (k2 = 1).
            topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1)
            dist_loss = torch.clamp(-topK_proto_dis + gamma2, min=0).mean()
            add_losses["dist_loss"] = dist_loss * self.cfg.MODEL.ROI_RELATION_HEAD.LOSS_WEIGHT_DIST

            # Prototype-based learning: pull relation to its gt prototype, push from negatives
            gamma = 3.0
            rel_rep_expand = rel_rep.unsqueeze(1).expand(-1, self.num_rel_cls, -1)
            predicate_proto_expand = predicate_proto.unsqueeze(0).expand(rel_labels.size(0), -1, -1)
            distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2
            mask_neg = torch.ones_like(distance_set)
            mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0
            distance_set_neg = distance_set * mask_neg
            distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]
            sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
            # Index 0 is the masked-to-zero gt; the next 10 entries are the hardest negatives (k1 = 10).
            topK_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10
            loss_sum = torch.clamp(distance_set_pos - topK_neg + gamma, min=0).mean()
            add_losses["loss_dis"] = loss_sum * self.cfg.MODEL.ROI_RELATION_HEAD.LOSS_WEIGHT_DIS

        ent_dists = ent_dists.split(num_objs, 0)
        rel_dists = rel_dists.split(num_rels, 0)

        # `add_data` slot matches the PE-NET base contract expected by ROIRelationHead.
        return ent_dists, rel_dists, add_losses, {}

    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed.weight

        assert proposals[0].mode == 'xyxy'

        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long()
            else:
                obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()

        return obj_dists, obj_preds

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for _ in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0

            obj_preds.append(out_label.long())
        return torch.cat(obj_preds, dim=0)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def fusion_func(x, y):
    return F.relu(x + y) - (x - y) ** 2


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
