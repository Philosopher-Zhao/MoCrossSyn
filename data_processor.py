import pandas as pd
import numpy as np
import torch
import os
import pickle
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, BRICS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

# ==================== 特征编码函数 ====================
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def safe_index(l, e):
    try:
        return l.index(e)
    except:
        return len(l)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set{allowable_set}")
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom, bool_id_feat=False, explicit_H=False, use_chirality=True):
    if bool_id_feat:
        return
    else:
        results = one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu',
                'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
            ]) + one_of_k_encoding(atom.GetDegree(),
                                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
                  one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2
                  ]) + [atom.GetIsAromatic()]
        if not explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]
        return np.array(results)

def bond_features(bond):
    """提取化学键特征"""
    bt = bond.GetBondType()
    bond_type_features = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC
    ]
    is_conjugated = bond.GetIsConjugated()
    is_in_ring = bond.IsInRing()
    features = np.array(bond_type_features + [is_conjugated, is_in_ring], dtype=np.float32)
    return features

# ==================== 分子图数据处理器 ====================
class MolecularGraphProcessor:
    """分子图数据处理器，增加Motif内部图构建"""
    def __init__(self, config):
        self.config = config
        self.atom_dim = config['data']['atom_dim']  # 78

    def brics_decompose_molecule(self, mol):
        try:
            fragments = list(BRICS.BRICSDecompose(mol))
            motif_mols = []
            for frag_smiles in fragments:
                frag_smiles_clean = frag_smiles.replace('[*]', '*')
                frag_mol = Chem.MolFromSmiles(frag_smiles_clean)
                if frag_mol is None:
                    frag_smiles_clean = frag_smiles.replace('[*]', '[H]')
                    frag_mol = Chem.MolFromSmiles(frag_smiles_clean)
                if frag_mol:
                    dummy_query = Chem.MolFromSmarts('[#0]')
                    if dummy_query is not None:
                        frag_mol_clean = Chem.DeleteSubstructs(frag_mol, dummy_query)
                        if frag_mol_clean is None or frag_mol_clean.GetNumAtoms() == 0:
                            continue
                        matches = mol.GetSubstructMatches(frag_mol_clean)
                        if matches:
                            motif_mols.append(frag_mol)
            return motif_mols
        except Exception as e:
            print(f"BRICS分解失败: {e}")
            return []

    def smiles_to_motif_graph(self, smiles):
        """将SMILES转换为Motif图，返回：节点特征、邻接矩阵、Motif内部图数据列表、原子节点数"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, None, 0

        try:
            atom_dim = self.atom_dim
            num_atoms = mol.GetNumAtoms()
            max_atoms_total = self.config['data']['max_atoms']

            # 原子特征列表
            atom_feat_list = [atom_features(atom, use_chirality=True) for atom in mol.GetAtoms()]

            # 获取Motif列表
            motifs = self.brics_decompose_molecule(mol)
            num_motifs = min(len(motifs), max_atoms_total - num_atoms)
            total_nodes = num_atoms + num_motifs

            # 初始化节点特征矩阵（原子部分）
            features = np.zeros((total_nodes, atom_dim))
            for i, feat in enumerate(atom_feat_list):
                features[i] = feat

            # 构建Motif内部图数据列表
            motif_graphs = []
            for motif in motifs[:num_motifs]:
                frag_mol = motif
                if frag_mol.GetNumAtoms() > 0:
                    motif_atom_feats = []
                    for atom in frag_mol.GetAtoms():
                        motif_atom_feats.append(atom_features(atom, use_chirality=True))
                    motif_edge_index = []
                    motif_edge_attr = []
                    for bond in frag_mol.GetBonds():
                        i = bond.GetBeginAtomIdx()
                        j = bond.GetEndAtomIdx()
                        motif_edge_index.append([i, j])
                        motif_edge_index.append([j, i])
                        feat = bond_features(bond)
                        motif_edge_attr.append(feat)
                        motif_edge_attr.append(feat)
                    motif_graphs.append({
                        'node_features': np.array(motif_atom_feats, dtype=np.float32),
                        'edge_index': np.array(motif_edge_index, dtype=np.int64).T,
                        'edge_attr': np.array(motif_edge_attr, dtype=np.float32)
                    })
                else:
                    motif_graphs.append(None)

            # 构建邻接矩阵
            adj_matrix = np.zeros((total_nodes, total_nodes))
            # 原子-原子键
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

            # 原子-Motif连接：遍历Motif，将其包含的原子与该Motif节点相连
            motif_atom_indices = []
            for motif in motifs[:num_motifs]:
                try:
                    matches = mol.GetSubstructMatches(motif)
                    indices_set = set()
                    for match in matches:
                        indices_set.update(match)
                    motif_atom_indices.append(list(indices_set))
                except:
                    motif_atom_indices.append([])

            for i, atom_indices in enumerate(motif_atom_indices):
                motif_idx = num_atoms + i
                for atom_idx in atom_indices:
                    adj_matrix[atom_idx, motif_idx] = 1
                    adj_matrix[motif_idx, atom_idx] = 1

            np.fill_diagonal(adj_matrix, 1)

            return features, adj_matrix, motif_graphs, num_atoms

        except Exception as e:
            print(f"处理SMILES {smiles} 时出错: {e}")
            return None, None, None, 0

    def process_drug_pairs(self, drug_smiles_file, drug_combinations_file, cell_features_file):
        print("处理药物对数据...")
        if not os.path.exists(drug_smiles_file) or not os.path.exists(drug_combinations_file) or not os.path.exists(cell_features_file):
            print("错误: 数据文件缺失")
            return []

        try:
            drug_smiles = pd.read_csv(drug_smiles_file)
            drug_combinations = pd.read_csv(drug_combinations_file)
            cell_features_df = pd.read_csv(cell_features_file, index_col='cell_line')

            if 'drugName' in drug_smiles.columns and 'SMILES' in drug_smiles.columns:
                drug_to_smiles = dict(zip(drug_smiles['drugName'], drug_smiles['SMILES']))
            elif 'drugNameOfficial' in drug_smiles.columns and 'smilesString' in drug_smiles.columns:
                drug_to_smiles = dict(zip(drug_smiles['drugNameOfficial'], drug_smiles['smilesString']))
            else:
                print("错误: 药物SMILES文件中缺少必要的列")
                return []

            processed_data = []
            skipped_count = 0

            for idx, row in drug_combinations.iterrows():
                if idx % 500 == 0:
                    print(f"处理进度: {idx}/{len(drug_combinations)}")

                drug1 = row['Drug1']
                drug2 = row['Drug2']
                cell_line = row['Cell line']

                if 'classification' in row:
                    classification = 1 if row['classification'] == 'synergy' else 0
                elif 'label' in row:
                    classification = row['label']
                else:
                    skipped_count += 1
                    continue

                if drug1 not in drug_to_smiles or drug2 not in drug_to_smiles:
                    skipped_count += 1
                    continue

                # 获取药物1的图数据及原子节点数
                features1, adj1, motif_graphs1, n1_atoms = self.smiles_to_motif_graph(drug_to_smiles[drug1])
                # 获取药物2的图数据及原子节点数
                features2, adj2, motif_graphs2, n2_atoms = self.smiles_to_motif_graph(drug_to_smiles[drug2])

                if features1 is None or features2 is None:
                    skipped_count += 1
                    continue

                # 合并两个药物的图
                combined_features = np.vstack([features1, features2])
                combined_adj = np.zeros((len(features1) + len(features2), len(features1) + len(features2)))
                combined_adj[0:len(features1), 0:len(features1)] = adj1
                combined_adj[len(features1):, len(features1):] = adj2

                if cell_line in cell_features_df.index:
                    cell_features = cell_features_df.loc[cell_line].values.astype(np.float32)
                else:
                    cell_features = np.zeros(self.config['data']['cell_feature_dim'], dtype=np.float32)

                processed_data.append({
                    'features': combined_features,
                    'adj_matrix': combined_adj,
                    'label': classification,
                    'drug1': drug1,
                    'drug2': drug2,
                    'cell_line': cell_line,
                    'cell_features': cell_features,
                    'drug1_nodes': n1_atoms,
                    'drug2_nodes': n2_atoms,
                    'motif_graphs_drug1': motif_graphs1,
                    'motif_graphs_drug2': motif_graphs2
                })

            print(f"数据处理完成: {len(processed_data)} 个样本成功, {skipped_count} 个跳过")
            return processed_data
        except Exception as e:
            print(f"处理药物对数据时出错: {e}")
            return []

    def split_and_save_data(self, processed_data, output_dir='data'):
        if not processed_data:
            print("错误: 没有可处理的数据")
            return None, None, None

        try:
            train_data, temp_data = train_test_split(
                processed_data, test_size=0.4, random_state=42,
                stratify=[d['label'] for d in processed_data]
            )
            val_data, test_data = train_test_split(
                temp_data, test_size=0.5, random_state=42,
                stratify=[d['label'] for d in temp_data]
            )
        except:
            train_data, temp_data = train_test_split(processed_data, test_size=0.4, random_state=42)
            val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        data_dir = os.path.join(output_dir, 'drug_synergy')
        os.makedirs(data_dir, exist_ok=True)

        def save_data(data, filename):
            data_dict = {
                'features': [d['features'] for d in data],
                'adj_matrix': [d['adj_matrix'] for d in data],
                'label': [d['label'] for d in data],
                'drug1': [d['drug1'] for d in data],
                'drug2': [d['drug2'] for d in data],
                'cell_line': [d['cell_line'] for d in data],
                'cell_features': [d['cell_features'] for d in data],
                'drug1_nodes': [d['drug1_nodes'] for d in data],  # 原子节点数
                'drug2_nodes': [d['drug2_nodes'] for d in data],  # 可选
                'motif_graphs_drug1': [d['motif_graphs_drug1'] for d in data],
                'motif_graphs_drug2': [d['motif_graphs_drug2'] for d in data]
            }
            with open(os.path.join(data_dir, filename), 'wb') as f:
                pickle.dump(data_dict, f)

        save_data(train_data, 'train.pkl')
        save_data(val_data, 'val.pkl')
        save_data(test_data, 'test.pkl')

        print(f"数据划分完成: 训练集 {len(train_data)}, 验证集 {len(val_data)}, 测试集 {len(test_data)}")
        return train_data, val_data, test_data
