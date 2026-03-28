import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
from config import CONFIG
from torch_geometric.nn import GINEConv, global_add_pool

# ==================== 原Transformer的数据预处理模块 ====================
class DataPreprocessor:
    def __init__(self, drug_smiles_path, drug_target_path, target_target_path,
                 drug_drug_path, target_features_path):
        """初始化数据预处理器"""
        self.drug_smiles_df = pd.read_csv(drug_smiles_path)
        self.drug_target_df = pd.read_csv(drug_target_path)
        self.target_target_df = pd.read_csv(target_target_path)
        self.drug_drug_df = pd.read_csv(drug_drug_path)
        self.target_features_df = pd.read_csv(target_features_path)

        self.drug_to_idx = {}
        self.target_to_idx = {}
        self.idx_to_drug = {}
        self.idx_to_target = {}
        self.drug_features = None
        self.target_features = None
        self.interaction_matrix = None
        self.drug_adj_matrix = None
        self.target_adj_matrix = None

    def smiles_to_ecfp(self, smiles, radius=2, n_bits=1024):
        """将SMILES转换为ECFP指纹"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)

    def preprocess_drug_features(self):
        """预处理药物特征：SMILES -> ECFP"""
        print("Processing drug features from SMILES...")
        drug_features = []
        drug_ids = []

        for _, row in self.drug_smiles_df.iterrows():
            smiles = row['smilesString']
            drug_id = row['drugNameOfficial']

            ecfp = self.smiles_to_ecfp(smiles)
            drug_features.append(ecfp)
            drug_ids.append(drug_id)

            if drug_id not in self.drug_to_idx:
                idx = len(self.drug_to_idx)
                self.drug_to_idx[drug_id] = idx
                self.idx_to_drug[idx] = drug_id

        self.drug_features = np.array(drug_features)
        print(f"Processed {len(self.drug_features)} drugs, feature shape: {self.drug_features.shape}")
        return self.drug_features, drug_ids

    def preprocess_target_features(self):
        """预处理靶点特征"""
        print("Processing target features...")
        feature_columns = [col for col in self.target_features_df.columns if col.startswith('Feature_')]
        target_features = self.target_features_df[feature_columns].values
        target_names = self.target_features_df['Target_Name'].values

        scaler = StandardScaler()
        target_features = scaler.fit_transform(target_features)

        for i, name in enumerate(target_names):
            if name not in self.target_to_idx:
                idx = len(self.target_to_idx)
                self.target_to_idx[name] = idx
                self.idx_to_target[idx] = name

        self.target_features = target_features
        print(f"Processed {len(target_features)} targets, feature shape: {target_features.shape}")
        return self.target_features, target_names

    def build_interaction_matrix(self):
        """构建药物-靶点相互作用矩阵"""
        print("Building drug-target interaction matrix...")
        n_drugs = len(self.drug_to_idx)
        n_targets = len(self.target_to_idx)

        interaction_matrix = np.zeros((n_drugs, n_targets), dtype=np.float32)

        for _, row in self.drug_target_df.iterrows():
            drug_name = row['normalized_name']
            target_name = row['target_name']

            if drug_name in self.drug_to_idx and target_name in self.target_to_idx:
                drug_idx = self.drug_to_idx[drug_name]
                target_idx = self.target_to_idx[target_name]
                interaction_matrix[drug_idx, target_idx] = 1.0

        self.interaction_matrix = interaction_matrix
        print(f"Interaction matrix shape: {interaction_matrix.shape}")
        return self.interaction_matrix

    def build_drug_drug_graph(self):
        """构建药物-药物相互作用图"""
        print("Building drug-drug interaction graph...")
        n_drugs = len(self.drug_to_idx)
        drug_adj_matrix = np.zeros((n_drugs, n_drugs), dtype=np.float32)

        for _, row in self.drug_drug_df.iterrows():
            drug1_id = row['药物1']
            drug2_id = row['药物2']

            if drug1_id in self.drug_to_idx and drug2_id in self.drug_to_idx:
                idx1 = self.drug_to_idx[drug1_id]
                idx2 = self.drug_to_idx[drug2_id]
                drug_adj_matrix[idx1, idx2] = 1.0
                drug_adj_matrix[idx2, idx1] = 1.0

        np.fill_diagonal(drug_adj_matrix, 1.0)
        self.drug_adj_matrix = drug_adj_matrix
        return self.drug_adj_matrix

    def build_target_target_graph(self):
        """构建靶点-靶点相互作用图"""
        print("Building target-target interaction graph...")
        n_targets = len(self.target_to_idx)
        target_adj_matrix = np.zeros((n_targets, n_targets), dtype=np.float32)

        for _, row in self.target_target_df.iterrows():
            target1 = row['node1']
            target2 = row['node2']
            score = row['combined_score']

            if target1 in self.target_to_idx and target2 in self.target_to_idx:
                idx1 = self.target_to_idx[target1]
                idx2 = self.target_to_idx[target2]
                target_adj_matrix[idx1, idx2] = score
                target_adj_matrix[idx2, idx1] = score

        np.fill_diagonal(target_adj_matrix, 1.0)
        self.target_adj_matrix = target_adj_matrix
        return self.target_adj_matrix

    def preprocess_all(self):
        """预处理所有数据"""
        self.preprocess_drug_features()
        self.preprocess_target_features()
        self.build_interaction_matrix()
        self.build_drug_drug_graph()
        self.build_target_target_graph()
        return self

# ==================== PyTorch版本的Transformer模块 ====================
class FeatureProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.projection(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, adjoin_matrix=None, dist_matrix=None):
        batch_size = q.size(0)
        seq_len = q.size(1)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = torch.tensor(self.depth, dtype=torch.float32, device=q.device)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 0, float('-inf'))

        if adjoin_matrix is not None:
            if adjoin_matrix.dim() == 3:
                adjoin_matrix = adjoin_matrix.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            elif adjoin_matrix.dim() == 4 and adjoin_matrix.size(1) != self.num_heads:
                adjoin_matrix = adjoin_matrix.mean(dim=1, keepdim=True).expand(-1, self.num_heads, -1, -1)
            if adjoin_matrix.size(-1) != scaled_attention_logits.size(-1):
                current = adjoin_matrix.size(-1)
                target = scaled_attention_logits.size(-1)
                if current < target:
                    pad = target - current
                    adjoin_matrix = F.pad(adjoin_matrix, (0, pad, 0, pad), "constant", 0)
                else:
                    adjoin_matrix = adjoin_matrix[:, :, :target, :target]
            scaled_attention_logits = scaled_attention_logits + adjoin_matrix

        if dist_matrix is not None:
            if dist_matrix.dim() == 3:
                dist_matrix = dist_matrix.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            elif dist_matrix.dim() == 4 and dist_matrix.size(1) != self.num_heads:
                dist_matrix = dist_matrix.mean(dim=1, keepdim=True).expand(-1, self.num_heads, -1, -1)
            if dist_matrix.size(-1) != scaled_attention_logits.size(-1):
                current = dist_matrix.size(-1)
                target = scaled_attention_logits.size(-1)
                if current < target:
                    pad = target - current
                    dist_matrix = F.pad(dist_matrix, (0, pad, 0, pad), "constant", 0)
                else:
                    dist_matrix = dist_matrix[:, :, :target, :target]
            scaled_attention_logits = scaled_attention_logits + dist_matrix

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.dense(output)
        return output, attention_weights

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha_local = MultiHeadAttention(d_model, num_heads)
        self.mha_global = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.GELU(),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(self, x, local_matrix=None, global_matrix=None):
        attn_local, _ = self.mha_local(x, x, x, adjoin_matrix=local_matrix)
        attn_local = self.dropout1(attn_local)
        out1 = self.layernorm1(x + attn_local)

        attn_global, _ = self.mha_global(out1, out1, out1, dist_matrix=global_matrix)
        attn_global = self.dropout2(attn_global)
        out2 = self.layernorm2(out1 + attn_global)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)
        return out3

class HeterogeneousEncoder(nn.Module):
    def __init__(self, drug_input_dim, target_input_dim, d_model=256,
                 num_heads=8, dff=512, num_layers=4, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.drug_projection = FeatureProjection(drug_input_dim, d_model)
        self.target_projection = FeatureProjection(target_input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(rate)
        self.drug_extractor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        # 新增：用于存储预计算的距离矩阵
        self.register_buffer('distance_matrix', None)

    def set_distance_matrix(self, distance_matrix):
        """注册距离矩阵，形状 [n_total, n_total]"""
        if not isinstance(distance_matrix, torch.Tensor):
            distance_matrix = torch.from_numpy(distance_matrix).float()
        self.register_buffer('distance_matrix', distance_matrix)

    def build_heterogeneous_graph(self, drug_adj, target_adj, interaction):
        batch_size = drug_adj.size(0)
        n_drugs = drug_adj.size(1)
        n_targets = target_adj.size(1)
        n_total = n_drugs + n_targets
        full_adj = torch.zeros(batch_size, n_total, n_total, device=drug_adj.device)
        full_adj[:, :n_drugs, :n_drugs] = drug_adj
        full_adj[:, n_drugs:, n_drugs:] = target_adj
        full_adj[:, :n_drugs, n_drugs:] = interaction
        full_adj[:, n_drugs:, :n_drugs] = interaction.transpose(1, 2)
        return full_adj

    def forward(self, drug_features, target_features, drug_adj_matrix,
                target_adj_matrix, interaction_matrix):
        batch_size = drug_features.size(0)
        drug_emb = self.drug_projection(drug_features)
        target_emb = self.target_projection(target_features)
        combined_emb = torch.cat([drug_emb, target_emb], dim=1)
        seq_len = combined_emb.size(1)
        if seq_len > self.pos_encoding.size(1):
            additional = torch.randn(1, seq_len - self.pos_encoding.size(1), self.d_model,
                                      device=self.pos_encoding.device)
            self.pos_encoding = nn.Parameter(torch.cat([self.pos_encoding.data, additional], dim=1))
        combined_emb = combined_emb + self.pos_encoding[:, :seq_len, :]
        combined_emb = self.dropout(combined_emb)

        # 局部邻接矩阵（结构偏置）
        local_matrix = self.build_heterogeneous_graph(
            drug_adj_matrix, target_adj_matrix, interaction_matrix
        )

        # 全局距离矩阵偏置（预计算）
        if self.distance_matrix is not None:
            # distance_matrix: [n_total, n_total] -> [batch, n_total, n_total]
            global_matrix = self.distance_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # 若未设置，则回退到零矩阵
            global_matrix = torch.zeros_like(local_matrix)

        # 确保在同一设备
        local_matrix = local_matrix.to(combined_emb.device)
        global_matrix = global_matrix.to(combined_emb.device)

        for enc_layer in self.enc_layers:
            combined_emb = enc_layer(combined_emb, local_matrix=local_matrix, global_matrix=global_matrix)

        n_drugs = drug_features.size(1)
        drug_output = combined_emb[:, :n_drugs, :]
        drug_output = self.drug_extractor(drug_output)
        return drug_output

# ==================== Motif 编码器 ====================
class MotifEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, out_dim=128):
        super().__init__()
        self.conv1 = GINEConv(nn=nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ), edge_dim=edge_dim)
        self.conv2 = GINEConv(nn=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ), edge_dim=edge_dim)
        self.pool = global_add_pool
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, batch_data):
        x = F.relu(self.conv1(batch_data.x, batch_data.edge_index, batch_data.edge_attr))
        x = F.relu(self.conv2(x, batch_data.edge_index, batch_data.edge_attr))
        motif_emb = self.pool(x, batch_data.batch)
        return self.fc(motif_emb)

# ==================== 新增的GAT层（替换GRU） ====================
class GATLayer(nn.Module):
    """基于MultiHeadAttention的图注意力层，用于替换原GRU模块"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, adj, x):
        """
        adj: [batch, num_nodes, num_nodes] 邻接矩阵（1表示有边，0表示无边）
        x: [batch, num_nodes, d_model] 节点特征
        """
        # 生成注意力掩码：有边的地方为0，无边的地方为 -inf
        attention_mask = (1 - adj) * -1e9  # [batch, num_nodes, num_nodes]
        attn_out, _ = self.mha(x, x, x, adjoin_matrix=attention_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

# ==================== 整合模型的组件 ====================
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.LayerNorm(hid_dim),
            nn.Dropout(p=dropout),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, embeds):
        return self.net(embeds)

class GRU(nn.Module):  # 保留原GRU类，但不再在blocks中使用（可能用于init）
    def __init__(self, in_dim, hid_dim, out_dim=None, dropout=0.3):
        super().__init__()
        self.out_dim = out_dim
        dim = in_dim if out_dim is None else out_dim
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hid_dim, dim),
            nn.Dropout(p=dropout)
        )
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(dim)

    def messaging(self, adjs, embeds):
        return torch.bmm(adjs, embeds)

    def forward(self, adjs, embeds):
        embeds = self.norm1(embeds + self.messaging(adjs, embeds))
        if self.out_dim is None:
            embeds = self.norm2(embeds + self.ffn(embeds))
        else:
            embeds = self.norm2(self.ffn(embeds))
        return embeds

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / torch.sqrt(torch.tensor(self.weight.size(1), dtype=torch.float))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, input):
        support = torch.bmm(input, self.weight.unsqueeze(0).expand(input.size(0), -1, -1))
        output = torch.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class CrossDrugAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.norm(attn_output)
        return attn_output, attn_weights

class DrugSynergyModel(nn.Module):
    def __init__(self, use_motif_encoder=True):
        super().__init__()
        data_config = CONFIG['data']
        model_config = CONFIG['model']
        train_config = CONFIG['train']

        self.num_block = model_config['block']['num']
        self.atom_hid_dim = data_config['atom_hid_dim']
        self.atom_dim = data_config['atom_dim']
        self.dropout_p = train_config['dropout']
        self.is_cat_after_readout = model_config['is_cat_after_readout']
        self.is_rg = model_config['is_rg']
        self.is_gcn = model_config['is_gcn']
        self.is_joint = model_config['is_joint']
        self.use_motif_encoder = use_motif_encoder
        if use_motif_encoder:
            bond_feat_dim = 6  # 由 bond_features 决定
            self.motif_encoder = MotifEncoder(
                node_dim=self.atom_dim,
                edge_dim=bond_feat_dim,
                hidden_dim=128,
                out_dim=self.atom_dim  # 输出维度为 atom_dim (78)
            )
        else:
            self.motif_encoder = None

        self.input_proj = nn.Sequential(
            nn.Linear(self.atom_dim, self.atom_hid_dim),
            nn.LayerNorm(self.atom_hid_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p)
        )

        self.cross_drug_attention = CrossDrugAttention(
            embed_dim=self.atom_hid_dim,
            num_heads=model_config['cross_attn_heads'],
            dropout=train_config['dropout']
        )

        if model_config['is_alpha_learn']:
            self.cross_attn_weight = nn.Parameter(torch.tensor(model_config['alpha']))
        else:
            self.cross_attn_weight = model_config['alpha']

        self.cross_attn_fc = nn.Sequential(
            nn.Linear(self.atom_hid_dim, self.atom_hid_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.atom_hid_dim, self.atom_hid_dim),
            nn.LayerNorm(self.atom_hid_dim)
        )

        cell_proj_config = model_config['cell_projection']
        cell_input_dim = data_config['cell_feature_dim']
        cell_layers = []
        input_dim = cell_input_dim
        for hidden_dim in cell_proj_config['hidden_dims']:
            cell_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_p)
            ])
            input_dim = hidden_dim
        self.cell_projection = nn.Sequential(*cell_layers)

        # 修改点：根据 is_gcn 决定使用 GCN 还是 GAT 层
        if self.is_gcn:
            self.blocks = nn.ModuleList([
                GraphConvolution(self.atom_hid_dim, self.atom_hid_dim) for _ in range(self.num_block)
            ])
        else:
            # 使用新的 GATLayer 替换原来的 GRU
            self.blocks = nn.ModuleList([
                GATLayer(self.atom_hid_dim, model_config['attn_heads'], dropout=self.dropout_p)
                for _ in range(self.num_block)
            ])

        if model_config['is_init_gru']:
            self.init = GRU(self.atom_dim, model_config['block']['ffn']['hidden_dim'],
                            self.atom_hid_dim, dropout=self.dropout_p)
        else:
            self.init = nn.Sequential(
                nn.Linear(self.atom_dim, self.atom_hid_dim),
                nn.LayerNorm(self.atom_hid_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p)
            )

        if self.is_rg:
            self.multi_head_attention = nn.MultiheadAttention(
                embed_dim=self.atom_hid_dim,
                num_heads=model_config['attn_heads'],
                dropout=train_config['dropout'],
                batch_first=True
            )
            if model_config['is_alpha_learn']:
                self.alpha = nn.Parameter(torch.tensor(model_config['alpha']))
            else:
                self.alpha = model_config['alpha']

        # 用于聚合多层输出的全连接层（仅在 is_cat_after_readout=True 时使用）
        if self.is_cat_after_readout:
            self.graph_output_dim_raw = self.atom_hid_dim * (self.num_block + 1)
            self.layer_fc = nn.Linear(self.graph_output_dim_raw, self.atom_hid_dim)
            self.graph_output_dim = self.atom_hid_dim
        else:
            self.graph_output_dim = self.atom_hid_dim

        self.cell_output_dim = model_config['cell_projection']['hidden_dims'][-1]

        # 修改点：添加用于双药特征融合的MLP（当 is_joint=False 时使用）
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * self.graph_output_dim, self.graph_output_dim),
            nn.ReLU(),
            nn.LayerNorm(self.graph_output_dim),
            nn.Dropout(self.dropout_p)
        )

        total_input_dim = self.graph_output_dim + self.cell_output_dim
        self.mlp = MLP(total_input_dim, model_config['mlp']['hidden_dim'],
                       1, dropout=self.dropout_p)

    def cal(self, embeds, adjs, masks):
        if embeds.size(-1) == self.atom_dim:
            if CONFIG['model']['is_init_gru']:
                embeds = self.init(adjs, embeds)
            else:
                embeds = self.init(embeds)

        history_embeds = [embeds]

        if self.is_rg:
            attn_output, r_adjs = self.multi_head_attention(
                embeds.detach(), embeds.detach(), embeds.detach(),
                key_padding_mask=masks
            )
            r_adjs = r_adjs.masked_fill(masks.unsqueeze(2).bool(), 0)
            r_adjs = r_adjs.masked_fill(masks.unsqueeze(1).bool(), 0)

            if CONFIG['model']['is_a_']:
                if CONFIG['model']['is_alpha_learn']:
                    alpha = torch.sigmoid(self.alpha)
                else:
                    alpha = self.alpha
                beta = 1.0 - alpha
                sum_adjs = beta * adjs + alpha * r_adjs
            else:
                sum_adjs = r_adjs
        else:
            sum_adjs = adjs

        for i, block in enumerate(self.blocks):
            # block 可能是 GraphConvolution 或 GATLayer，两者都接受 (adj, embeds)
            embeds = block(sum_adjs, embeds)
            history_embeds.append(embeds)

        if self.is_cat_after_readout:
            combined = torch.cat(history_embeds, dim=-1)
            embeds = self.layer_fc(combined)
        else:
            embeds = sum(history_embeds)

        embeds = embeds.masked_fill(masks.unsqueeze(2).bool(), 0)
        embeds = torch.sum(embeds, dim=1)
        return embeds

    def build_joint_graph_with_attention(self, embeds, adjs, masks, drug1_nodes):
        batch_size, num_nodes, _ = embeds.shape
        proj_embeds = self.input_proj(embeds)

        cross_attn_mask = torch.zeros((batch_size, num_nodes, num_nodes), device=embeds.device)
        for i in range(batch_size):
            d1 = min(drug1_nodes[i].item(), num_nodes)
            cross_attn_mask[i, :d1, d1:] = 1
            cross_attn_mask[i, d1:, :d1] = 1

        masks_bool = masks.bool()
        padding_mask_2d = masks_bool.unsqueeze(1) | masks_bool.unsqueeze(2)
        cross_attn_mask = cross_attn_mask * (~padding_mask_2d).float()

        attn_output, attn_weights = self.cross_drug_attention(proj_embeds, proj_embeds, proj_embeds)
        if len(attn_weights.shape) == 4:
            attn_weights = attn_weights.mean(dim=1)

        cross_attn_weights = attn_weights * cross_attn_mask
        if cross_attn_weights.sum() > 0:
            cross_attn_weights = F.softmax(cross_attn_weights, dim=-1)

        attn_enhanced_embeds = self.cross_attn_fc(attn_output)

        if isinstance(self.cross_attn_weight, torch.Tensor):
            weight = torch.sigmoid(self.cross_attn_weight)
        else:
            weight = self.cross_attn_weight

        joint_adjs = adjs + weight * cross_attn_weights
        row_sum = joint_adjs.sum(dim=-1, keepdim=True) + 1e-8
        joint_adjs = joint_adjs / row_sum

        return attn_enhanced_embeds, joint_adjs

    def forward(self, embeds, adjs, masks, cnn_masks, targets,
                cell_features, drug1_nodes,
                motif_batch_drug1=None, motif_batch_idx_drug1=None,
                motif_batch_drug2=None, motif_batch_idx_drug2=None):
        batch_size = embeds.size(0)
        device = embeds.device

        # 使用 Motif 编码器更新 embeds 中的 Motif 节点特征
        if self.use_motif_encoder and self.motif_encoder is not None:
            if motif_batch_drug1 is not None:
                motif_batch_drug1 = motif_batch_drug1.to(device)
                motif_emb_drug1 = self.motif_encoder(motif_batch_drug1)

                if motif_batch_idx_drug1 is not None:
                    motif_batch_idx_drug1 = motif_batch_idx_drug1.to(device)
                    motif_counts_drug1 = torch.bincount(motif_batch_idx_drug1, minlength=batch_size)
                else:
                    motif_counts_drug1 = torch.zeros(batch_size, dtype=torch.long, device=device)

                motif_counter = 0
                for i in range(batch_size):
                    num_motifs_i = motif_counts_drug1[i].item()
                    if num_motifs_i > 0:
                        start = drug1_nodes[i].item()
                        embeds[i, start:start+num_motifs_i, :] = motif_emb_drug1[motif_counter:motif_counter+num_motifs_i]
                        motif_counter += num_motifs_i

            if motif_batch_drug2 is not None:
                motif_batch_drug2 = motif_batch_drug2.to(device)
                motif_emb_drug2 = self.motif_encoder(motif_batch_drug2)

                if motif_batch_idx_drug2 is not None:
                    motif_batch_idx_drug2 = motif_batch_idx_drug2.to(device)
                    motif_counts_drug2 = torch.bincount(motif_batch_idx_drug2, minlength=batch_size)
                else:
                    motif_counts_drug2 = torch.zeros(batch_size, dtype=torch.long, device=device)

                if motif_batch_idx_drug1 is not None:
                    motif_counts_drug1 = torch.bincount(motif_batch_idx_drug1, minlength=batch_size)
                else:
                    motif_counts_drug1 = torch.zeros(batch_size, dtype=torch.long, device=device)

                motif_counter = 0
                for i in range(batch_size):
                    num_motifs_i = motif_counts_drug2[i].item()
                    if num_motifs_i > 0:
                        offset = drug1_nodes[i].item() + motif_counts_drug1[i].item()
                        embeds[i, offset:offset+num_motifs_i, :] = motif_emb_drug2[motif_counter:motif_counter+num_motifs_i]
                        motif_counter += num_motifs_i

        # 图特征提取
        if self.is_joint:
            attn_embeds, joint_adjs = self.build_joint_graph_with_attention(embeds, adjs, masks, drug1_nodes)
            graph_features = self.cal(attn_embeds, joint_adjs, masks)
        else:
            proj_embeds = self.input_proj(embeds)
            drug1_size = int(drug1_nodes.float().mean().item())
            embeds1 = self.cal(proj_embeds[:, :drug1_size, :],
                               adjs[:, :drug1_size, :drug1_size],
                               masks[:, :drug1_size])
            embeds2 = self.cal(proj_embeds[:, drug1_size:, :],
                               adjs[:, drug1_size:, drug1_size:],
                               masks[:, drug1_size:])

            # 修改点：将相加改为拼接+MLP
            combined = torch.cat([embeds1, embeds2], dim=1)  # [batch, 2*graph_output_dim]
            graph_features = self.fusion_mlp(combined)      # [batch, graph_output_dim]

        cell_projected = self.cell_projection(cell_features)
        combined_features = torch.cat([graph_features, cell_projected], dim=1)

        scores = self.mlp(combined_features)

        if CONFIG['data']['is_binary']:
            if scores.dim() == 1:
                scores = scores.unsqueeze(1)
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            loss = F.binary_cross_entropy_with_logits(scores, targets.float(), reduction='mean')
        else:
            loss = F.cross_entropy(scores, targets)

        return scores, loss, graph_features

class IntegratedDrugSynergyModel(nn.Module):
    def __init__(self, transformer_config=None):
        super().__init__()
        data_config = CONFIG['data']
        model_config = CONFIG['model']
        train_config = CONFIG['train']

        # GCN模块
        self.gcn_model = DrugSynergyModel(use_motif_encoder=True)
        self.gcn_feature_dim = self.gcn_model.graph_output_dim

        # Transformer模块
        if transformer_config is None:
            transformer_config = CONFIG['transformer']
        self.transformer_encoder = HeterogeneousEncoder(
            drug_input_dim=transformer_config['drug_input_dim'],
            target_input_dim=transformer_config['target_input_dim'],
            d_model=transformer_config['d_model'],
            num_heads=transformer_config['num_heads'],
            dff=transformer_config['dff'],
            num_layers=transformer_config['num_layers'],
            rate=transformer_config['dropout_rate']
        )
        self.transformer_feature_dim = transformer_config['d_model']

        # 细胞系特征处理（复用GCN的cell_projection）
        self.cell_projection = self.gcn_model.cell_projection
        cell_output_dim = self.gcn_model.cell_output_dim

        total_input_dim = (self.transformer_feature_dim * 2) + self.gcn_feature_dim + cell_output_dim
        self.final_mlp = MLP(
            total_input_dim,
            model_config['mlp']['hidden_dim'],
            1,
            dropout=train_config['dropout']
        )

        self.transformer_initialized = False
        self.all_transformer_features = None
        self.drug_id_to_idx = None

    def initialize_transformer(self, heterogeneous_data):
        device = next(self.parameters()).device
        drug_features = torch.FloatTensor(heterogeneous_data['drug_features']).unsqueeze(0).to(device)
        target_features = torch.FloatTensor(heterogeneous_data['target_features']).unsqueeze(0).to(device)
        drug_adj_matrix = torch.FloatTensor(heterogeneous_data['drug_adj_matrix']).unsqueeze(0).to(device)
        target_adj_matrix = torch.FloatTensor(heterogeneous_data['target_adj_matrix']).unsqueeze(0).to(device)
        interaction_matrix = torch.FloatTensor(heterogeneous_data['interaction_matrix']).unsqueeze(0).to(device)

        with torch.no_grad():
            all_drug_features = self.transformer_encoder(
                drug_features, target_features,
                drug_adj_matrix, target_adj_matrix,
                interaction_matrix
            )
        self.all_transformer_features = all_drug_features.squeeze(0)
        self.drug_id_to_idx = heterogeneous_data['drug_id_to_idx']

        # ---------- 新增：计算距离矩阵并注册到 Transformer 编码器 ----------
        import numpy as np
        from scipy.sparse.csgraph import shortest_path

        # 从 heterogeneous_data 中获取原始 numpy 矩阵
        drug_adj = heterogeneous_data['drug_adj_matrix']           # [n_drugs, n_drugs]
        target_adj = heterogeneous_data['target_adj_matrix']       # [n_targets, n_targets]
        inter = heterogeneous_data['interaction_matrix']           # [n_drugs, n_targets]

        n_drugs = drug_adj.shape[0]
        n_targets = target_adj.shape[0]
        n_total = n_drugs + n_targets

        # 构建完整邻接矩阵（与 build_heterogeneous_graph 一致）
        full_adj = np.zeros((n_total, n_total), dtype=np.float32)
        full_adj[:n_drugs, :n_drugs] = drug_adj
        full_adj[n_drugs:, n_drugs:] = target_adj
        full_adj[:n_drugs, n_drugs:] = inter
        full_adj[n_drugs:, :n_drugs] = inter.T

        # 计算最短路径距离（无权图，边权为1）
        dist_matrix = shortest_path(full_adj, directed=False, unweighted=True)
        # 将无穷大（不相通）替换为 0
        dist_matrix = np.where(np.isinf(dist_matrix), 0, dist_matrix)

        # 注册距离矩阵到 transformer_encoder
        self.transformer_encoder.set_distance_matrix(dist_matrix)
        # ----------------------------------------------------------------

        self.transformer_initialized = True

    def get_transformer_features(self, drug1_idx, drug2_idx):
        if not self.transformer_initialized:
            raise ValueError("Transformer模块未初始化")
        if drug1_idx < 0 or drug1_idx >= len(self.all_transformer_features) or \
           drug2_idx < 0 or drug2_idx >= len(self.all_transformer_features):
            device = self.all_transformer_features.device
            return torch.zeros(1, self.transformer_feature_dim * 2, device=device)
        drug1_feat = self.all_transformer_features[drug1_idx].unsqueeze(0)
        drug2_feat = self.all_transformer_features[drug2_idx].unsqueeze(0)
        return torch.cat([drug1_feat, drug2_feat], dim=-1)

    def forward(self, embeds, adjs, masks, cnn_masks, targets,
                cell_features, drug1_nodes, drug1_indices, drug2_indices,
                motif_batch_drug1=None, motif_batch_idx_drug1=None,
                motif_batch_drug2=None, motif_batch_idx_drug2=None):
        device = embeds.device

        # GCN模块
        gcn_scores, gcn_loss, gcn_features = self.gcn_model(
            embeds, adjs, masks, cnn_masks, targets,
            cell_features, drug1_nodes,
            motif_batch_drug1, motif_batch_idx_drug1,
            motif_batch_drug2, motif_batch_idx_drug2
        )

        # Transformer模块
        if drug1_indices.dim() == 0:
            drug1_indices = drug1_indices.unsqueeze(0)
        if drug2_indices.dim() == 0:
            drug2_indices = drug2_indices.unsqueeze(0)

        batch_size = len(drug1_indices)
        transformer_features_list = []
        for i in range(batch_size):
            d1 = drug1_indices[i].item()
            d2 = drug2_indices[i].item()
            tf_feat = self.get_transformer_features(d1, d2)
            transformer_features_list.append(tf_feat)
        transformer_features = torch.cat(transformer_features_list, dim=0)

        # 细胞系特征
        cell_projected = self.cell_projection(cell_features)

        # 拼接
        combined = torch.cat([transformer_features, gcn_features, cell_projected], dim=1)
        final_scores = self.final_mlp(combined)

        if CONFIG['data']['is_binary']:
            if final_scores.dim() == 1:
                final_scores = final_scores.unsqueeze(1)
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            loss = F.binary_cross_entropy_with_logits(final_scores, targets.float(), reduction='mean')
        else:
            loss = F.cross_entropy(final_scores, targets)

        return final_scores, loss, {'total_loss': loss.item()}