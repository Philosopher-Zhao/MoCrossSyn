import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from config import CONFIG
from model import DataPreprocessor

class IntegratedDrugSynergyDataset(Dataset):
    def __init__(self, data_path, heterogeneous_processor=None):
        super().__init__()
        self.data_path = data_path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.heterogeneous_processor = heterogeneous_processor
        print(f"加载数据集: {data_path}, 样本数: {len(self.data['label'])}")

        required_keys = ['features', 'adj_matrix', 'label', 'cell_features', 'drug1', 'drug2', 'drug1_nodes']
        for key in required_keys:
            if key not in self.data:
                raise KeyError(f"数据中缺少必要的键: {key}")

    def __getitem__(self, idx):
        item = {
            'features': self.data['features'][idx],
            'adj_matrix': self.data['adj_matrix'][idx],
            'label': self.data['label'][idx],
            'drug1': self.data['drug1'][idx],
            'drug2': self.data['drug2'][idx],
            'cell_line': self.data.get('cell_line', [''])[idx] if 'cell_line' in self.data else '',
            'cell_features': self.data['cell_features'][idx],
            'drug1_nodes': self.data['drug1_nodes'][idx],
            'motif_graphs_drug1': self.data.get('motif_graphs_drug1', [])[idx] if 'motif_graphs_drug1' in self.data else [],
            'motif_graphs_drug2': self.data.get('motif_graphs_drug2', [])[idx] if 'motif_graphs_drug2' in self.data else []
        }

        if self.heterogeneous_processor:
            drug1_name = item['drug1']
            drug2_name = item['drug2']
            item['drug1_idx'] = self.heterogeneous_processor.drug_to_idx.get(drug1_name, -1)
            item['drug2_idx'] = self.heterogeneous_processor.drug_to_idx.get(drug2_name, -1)
        else:
            item['drug1_idx'] = -1
            item['drug2_idx'] = -1

        return item

    def __len__(self):
        return len(self.data['label'])

def integrated_collate_fn(batch):
    batch_size = len(batch)
    max_atoms = CONFIG['data']['max_atoms']
    atom_dim = CONFIG['data']['atom_dim']
    cell_feature_dim = CONFIG['data']['cell_feature_dim']

    embeds_batch = torch.zeros(batch_size, max_atoms, atom_dim)
    adjs_batch = torch.zeros(batch_size, max_atoms, max_atoms)
    masks_batch = torch.ones(batch_size, max_atoms)
    cnn_masks_batch = torch.zeros(batch_size, max_atoms, max_atoms)
    targets_batch = []
    cell_features_batch = torch.zeros(batch_size, cell_feature_dim)
    drug1_nodes_batch = torch.zeros(batch_size, dtype=torch.long)
    drug1_indices_batch = torch.zeros(batch_size, dtype=torch.long)
    drug2_indices_batch = torch.zeros(batch_size, dtype=torch.long)
    drug1_names_batch = []
    drug2_names_batch = []
    cell_line_names_batch = []

    # 收集Motif图数据
    motif_graphs_drug1_all = []
    motif_graphs_drug2_all = []
    motif_batch_idx_drug1 = []
    motif_batch_idx_drug2 = []

    for i, sample in enumerate(batch):
        features = sample['features']
        adj_matrix = sample['adj_matrix']
        label = sample['label']
        cell_features = sample['cell_features']
        drug1_nodes = sample['drug1_nodes']
        drug1_idx = sample.get('drug1_idx', -1)
        drug2_idx = sample.get('drug2_idx', -1)
        drug1_name = sample['drug1']
        drug2_name = sample['drug2']
        cell_line_name = sample['cell_line']

        num_atoms = min(len(features), max_atoms)
        embeds_batch[i, :num_atoms] = torch.tensor(features[:num_atoms], dtype=torch.float)
        adjs_batch[i, :num_atoms, :num_atoms] = torch.tensor(adj_matrix[:num_atoms, :num_atoms], dtype=torch.float)
        masks_batch[i, :num_atoms] = 0
        cnn_masks_batch[i, :num_atoms, :num_atoms] = 1

        targets_batch.append(label)
        cell_features_batch[i] = torch.tensor(cell_features, dtype=torch.float)
        drug1_nodes_batch[i] = min(drug1_nodes, max_atoms)
        drug1_indices_batch[i] = drug1_idx
        drug2_indices_batch[i] = drug2_idx
        drug1_names_batch.append(drug1_name)
        drug2_names_batch.append(drug2_name)
        cell_line_names_batch.append(cell_line_name)

        # 处理 drug1 的 Motif 图
        for graph_data in sample['motif_graphs_drug1']:
            if graph_data is not None:
                data = Data(
                    x=torch.tensor(graph_data['node_features'], dtype=torch.float),
                    edge_index=torch.tensor(graph_data['edge_index'], dtype=torch.long),
                    edge_attr=torch.tensor(graph_data['edge_attr'], dtype=torch.float)
                )
                motif_graphs_drug1_all.append(data)
                motif_batch_idx_drug1.append(i)

        # 处理 drug2 的 Motif 图
        for graph_data in sample['motif_graphs_drug2']:
            if graph_data is not None:
                data = Data(
                    x=torch.tensor(graph_data['node_features'], dtype=torch.float),
                    edge_index=torch.tensor(graph_data['edge_index'], dtype=torch.long),
                    edge_attr=torch.tensor(graph_data['edge_attr'], dtype=torch.float)
                )
                motif_graphs_drug2_all.append(data)
                motif_batch_idx_drug2.append(i)

    if CONFIG['data']['is_binary']:
        targets_batch = torch.tensor(targets_batch, dtype=torch.float)
        if targets_batch.dim() == 1:
            targets_batch = targets_batch.unsqueeze(1)
    else:
        targets_batch = torch.tensor(targets_batch, dtype=torch.long)

    # 将 Motif 图列表打包为 Batch
    motif_batch_drug1 = Batch.from_data_list(motif_graphs_drug1_all) if motif_graphs_drug1_all else None
    motif_batch_drug2 = Batch.from_data_list(motif_graphs_drug2_all) if motif_graphs_drug2_all else None
    motif_batch_idx_drug1 = torch.tensor(motif_batch_idx_drug1, dtype=torch.long) if motif_batch_idx_drug1 else None
    motif_batch_idx_drug2 = torch.tensor(motif_batch_idx_drug2, dtype=torch.long) if motif_batch_idx_drug2 else None

    return (
        embeds_batch, adjs_batch, masks_batch, cnn_masks_batch,
        targets_batch, cell_features_batch, drug1_nodes_batch,
        drug1_indices_batch, drug2_indices_batch,
        drug1_names_batch, drug2_names_batch, cell_line_names_batch,
        motif_batch_drug1, motif_batch_idx_drug1,
        motif_batch_drug2, motif_batch_idx_drug2
    )

class HeterogeneousDataLoader:
    def __init__(self, config):
        self.config = config
        self.preprocessor = None

    def load_heterogeneous_data(self):
        print("加载异构网络数据...")
        required_files = [
            self.config['paths']['drug_smiles'],
            self.config['paths']['drug_target_interactions'],
            self.config['paths']['target_target_interactions'],
            self.config['paths']['drug_drug_interactions'],
            self.config['paths']['target_features']
        ]
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"警告: 文件 {file_path} 不存在")
                return None

        try:
            self.preprocessor = DataPreprocessor(
                drug_smiles_path=self.config['paths']['drug_smiles'],
                drug_target_path=self.config['paths']['drug_target_interactions'],
                target_target_path=self.config['paths']['target_target_interactions'],
                drug_drug_path=self.config['paths']['drug_drug_interactions'],
                target_features_path=self.config['paths']['target_features']
            )
            self.preprocessor.preprocess_all()
            heterogeneous_data = {
                'drug_features': self.preprocessor.drug_features,
                'target_features': self.preprocessor.target_features,
                'interaction_matrix': self.preprocessor.interaction_matrix,
                'drug_adj_matrix': self.preprocessor.drug_adj_matrix,
                'target_adj_matrix': self.preprocessor.target_adj_matrix,
                'drug_ids': list(self.preprocessor.drug_to_idx.keys()),
                'drug_id_to_idx': self.preprocessor.drug_to_idx
            }
            print(f"异构网络数据加载完成: 药物 {len(self.preprocessor.drug_to_idx)}, 靶点 {len(self.preprocessor.target_to_idx)}")
            return heterogeneous_data
        except Exception as e:
            print(f"加载异构网络数据时出错: {e}")
            return None

    def get_data_loaders(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(self.config['paths']['output_dir'], 'drug_synergy')

        heterogeneous_data = self.load_heterogeneous_data()
        if heterogeneous_data is None:
            print("错误: 无法加载异构网络数据")
            return None, None, None, None

        data_files = ['train.pkl', 'val.pkl', 'test.pkl']
        missing_files = []
        for f in data_files:
            file_path = os.path.join(data_dir, f)
            if not os.path.exists(file_path):
                missing_files.append(file_path)

        if missing_files:
            print("警告: 以下药物协同数据文件不存在，尝试生成...")
            from data_processor import MolecularGraphProcessor
            processor = MolecularGraphProcessor(self.config)
            processed_data = processor.process_drug_pairs(
                self.config['paths']['drug_smiles'],
                self.config['paths']['drug_combinations'],
                self.config['paths']['cell_features']
            )
            if not processed_data:
                print("错误: 无法处理药物对数据")
                return None, None, None, None
            processor.split_and_save_data(processed_data, self.config['paths']['output_dir'])

        try:
            train_dataset = IntegratedDrugSynergyDataset(
                os.path.join(data_dir, 'train.pkl'),
                self.preprocessor
            )
            val_dataset = IntegratedDrugSynergyDataset(
                os.path.join(data_dir, 'val.pkl'),
                self.preprocessor
            )
            test_dataset = IntegratedDrugSynergyDataset(
                os.path.join(data_dir, 'test.pkl'),
                self.preprocessor
            )
        except Exception as e:
            print(f"创建数据集时出错: {e}")
            return None, None, None, None

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['train']['batch_size'],
            shuffle=True,
            collate_fn=integrated_collate_fn,
            num_workers=self.config['train']['num_workers'],
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['train']['batch_size'],
            shuffle=False,
            collate_fn=integrated_collate_fn,
            num_workers=self.config['train']['num_workers']
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['train']['batch_size'],
            shuffle=False,
            collate_fn=integrated_collate_fn,
            num_workers=self.config['train']['num_workers']
        )

        print(f"数据加载器创建完成: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}, 测试集 {len(test_dataset)}")
        return train_loader, val_loader, test_loader, heterogeneous_data
