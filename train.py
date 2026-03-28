'''
import os
import pickle
import torch
import torch.optim as optim
import numpy as np
import time
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from config import CONFIG
from dataset import HeterogeneousDataLoader
from model import IntegratedDrugSynergyModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, path, epoch=None, metrics=None):
    state = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'metrics': metrics
    }
    torch.save(state, path)
    print(f"模型保存到: {path}")

def load_model(model, path):
    if not os.path.exists(path):
        print(f"模型文件不存在: {path}")
        return None
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    print(f"模型加载自 {path} (epoch: {state.get('epoch', 'N/A')})")
    return state.get('metrics', {})

from torch_geometric.data import Batch

def move_data_to_device(data, device):
    (embeds, adjs, masks, cnn_masks, targets,
     cell_features, drug1_nodes, drug1_indices, drug2_indices,
     drug1_names, drug2_names, cell_line_names,
     motif_batch_drug1, motif_batch_idx_drug1,
     motif_batch_drug2, motif_batch_idx_drug2) = data

    embeds = embeds.to(device)
    adjs = adjs.to(device)
    masks = masks.to(device)
    cnn_masks = cnn_masks.to(device)

    if CONFIG['data']['is_binary']:
        targets = targets.to(device).float()
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
    else:
        targets = targets.to(device).long()

    cell_features = cell_features.to(device)
    drug1_nodes = drug1_nodes.to(device)
    drug1_indices = drug1_indices.to(device)
    drug2_indices = drug2_indices.to(device)

    # 处理 motif_batch_drug1：如果是列表，先转换为 Batch 并移到设备
    if motif_batch_drug1 is not None:
        if isinstance(motif_batch_drug1, list):
            motif_batch_drug1 = [d.to(device) for d in motif_batch_drug1]
            motif_batch_drug1 = Batch.from_data_list(motif_batch_drug1)
        else:
            motif_batch_drug1 = motif_batch_drug1.to(device)
    if motif_batch_idx_drug1 is not None:
        motif_batch_idx_drug1 = motif_batch_idx_drug1.to(device)

    if motif_batch_drug2 is not None:
        if isinstance(motif_batch_drug2, list):
            motif_batch_drug2 = [d.to(device) for d in motif_batch_drug2]
            motif_batch_drug2 = Batch.from_data_list(motif_batch_drug2)
        else:
            motif_batch_drug2 = motif_batch_drug2.to(device)
    if motif_batch_idx_drug2 is not None:
        motif_batch_idx_drug2 = motif_batch_idx_drug2.to(device)

    return (embeds, adjs, masks, cnn_masks, targets,
            cell_features, drug1_nodes, drug1_indices, drug2_indices,
            drug1_names, drug2_names, cell_line_names,
            motif_batch_drug1, motif_batch_idx_drug1,
            motif_batch_drug2, motif_batch_idx_drug2)

def calculate_metrics(y_true, y_pred, y_scores=None):
    if y_true.dim() > 1:
        y_true = y_true.squeeze()
    if y_pred.dim() > 1:
        y_pred = y_pred.squeeze()

    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    metrics = {}
    try:
        metrics['acc'] = accuracy_score(y_true_np, y_pred_np)
        metrics['f1'] = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        metrics['precision'] = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        metrics['recall'] = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)

        if y_scores is not None and len(np.unique(y_true_np)) > 1:
            if CONFIG['data']['is_binary']:
                if y_scores.dim() > 1:
                    y_scores = y_scores.squeeze(1)
                y_probs = torch.sigmoid(y_scores).cpu().numpy()
                metrics['auc'] = roc_auc_score(y_true_np, y_probs)
            else:
                y_probs = torch.softmax(y_scores, dim=1).cpu().numpy()
                metrics['auc'] = roc_auc_score(y_true_np, y_probs, multi_class='ovr')
    except Exception as e:
        print(f"计算指标时出错: {e}")
        metrics = {'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0}
    return metrics

def train_epoch(model, dataloader, optimizer, scheduler=None):
    model.train()
    total_loss = 0
    all_true = []
    all_pred = []
    all_scores = []

    for batch in dataloader:
        data = move_data_to_device(batch, device)
        model_inputs = (
            data[0], data[1], data[2], data[3], data[4],
            data[5], data[6], data[7], data[8],
            data[12], data[13], data[14], data[15]
        )
        scores, loss, loss_dict = model(*model_inputs)

        optimizer.zero_grad()
        loss.backward()
        if CONFIG['train']['grad_clip']['enabled']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['train']['grad_clip']['max_norm'])
        optimizer.step()

        total_loss += loss.item()

        if CONFIG['data']['is_binary']:
            if scores.dim() > 1:
                scores_flat = scores.squeeze(1)
            else:
                scores_flat = scores
            preds = (torch.sigmoid(scores_flat) > 0.5).long()
        else:
            preds = torch.argmax(scores, dim=1)

        if preds.dim() == 0:
            preds = preds.unsqueeze(0)

        targets_for_metrics = data[4]
        if targets_for_metrics.dim() > 1:
            targets_for_metrics = targets_for_metrics.squeeze(1)

        all_true.append(targets_for_metrics.cpu())
        all_pred.append(preds.cpu())
        all_scores.append(scores.detach().cpu())

    if scheduler is not None and CONFIG['train']['scheduler']['type'] != 'ReduceLROnPlateau':
        scheduler.step()

    all_true = torch.cat(all_true)
    all_pred = torch.cat(all_pred)
    all_scores = torch.cat(all_scores)

    metrics = calculate_metrics(all_true, all_pred, all_scores)
    metrics['loss'] = total_loss / len(dataloader)
    return metrics

def validate(model, dataloader, return_predictions=False):
    model.eval()
    total_loss = 0
    all_true = []
    all_pred = []
    all_scores = []
    all_drug1_names = []
    all_drug2_names = []
    all_cell_line_names = []

    with torch.no_grad():
        for batch in dataloader:
            data = move_data_to_device(batch, device)
            model_inputs = (
                data[0], data[1], data[2], data[3], data[4],
                data[5], data[6], data[7], data[8],
                data[12], data[13], data[14], data[15]
            )
            scores, loss, loss_dict = model(*model_inputs)

            total_loss += loss.item()

            if CONFIG['data']['is_binary']:
                if scores.dim() > 1:
                    scores_flat = scores.squeeze(1)
                else:
                    scores_flat = scores
                preds = (torch.sigmoid(scores_flat) > 0.5).long()
            else:
                preds = torch.argmax(scores, dim=1)

            if preds.dim() == 0:
                preds = preds.unsqueeze(0)

            targets_for_metrics = data[4]
            if targets_for_metrics.dim() > 1:
                targets_for_metrics = targets_for_metrics.squeeze(1)

            all_true.append(targets_for_metrics.cpu())
            all_pred.append(preds.cpu())
            all_scores.append(scores.cpu())

            all_drug1_names.extend(data[9])
            all_drug2_names.extend(data[10])
            all_cell_line_names.extend(data[11])

    all_true = torch.cat(all_true)
    all_pred = torch.cat(all_pred)
    all_scores = torch.cat(all_scores)
    metrics = calculate_metrics(all_true, all_pred, all_scores)
    metrics['loss'] = total_loss / len(dataloader)

    if return_predictions:
        if CONFIG['data']['is_binary']:
            scores_np = all_scores.cpu().numpy()
            if scores_np.ndim > 1:
                scores_np = scores_np.squeeze(1)
            probs_np = torch.sigmoid(all_scores).cpu().numpy()
            if probs_np.ndim > 1:
                probs_np = probs_np.squeeze(1)
        else:
            scores_np = all_scores.cpu().numpy().tolist()
            probs_np = torch.softmax(all_scores, dim=1).cpu().numpy().tolist()

        predictions_df = pd.DataFrame({
            'drug1': all_drug1_names,
            'drug2': all_drug2_names,
            'cell_line': all_cell_line_names,
            'true_label': all_true.cpu().numpy(),
            'predicted_label': all_pred.cpu().numpy(),
            'prediction_score': scores_np,
            'predicted_probability': probs_np
        })
        return metrics, predictions_df
    else:
        return metrics, all_true, all_pred, all_scores

# ========== 修改后的特征提取函数，支持原始特征与学习后特征 ==========
def extract_features_and_save(model, dataloader, device, save_path, heterogeneous_data=None, use_learned=True):
    """
    提取特征并保存。
    Args:
        model: 模型实例
        dataloader: 数据加载器（训练集）
        device: 设备
        save_path: 保存路径
        heterogeneous_data: 异构网络数据（包含 drug_features 等），当 use_learned=False 时需要
        use_learned: True 表示提取经过模型学习后的特征（训练后）；False 表示提取原始输入特征（训练前）
    """
    model.eval()
    all_features = []
    all_drug1_names = []
    all_drug2_names = []
    all_cell_line_names = []
    all_true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            data = move_data_to_device(batch, device)
            embeds, adjs, masks, cnn_masks, targets, cell_features, drug1_nodes, drug1_indices, drug2_indices = data[:9]
            drug1_names, drug2_names, cell_line_names = data[9], data[10], data[11]
            motif_batch_drug1, motif_batch_idx_drug1, motif_batch_drug2, motif_batch_idx_drug2 = data[12:16]

            batch_size = embeds.size(0)

            if use_learned:
                # ===== 学习后的特征（训练后） =====
                # GCN 特征
                _, _, gcn_features = model.gcn_model(
                    embeds, adjs, masks, cnn_masks, targets,
                    cell_features, drug1_nodes,
                    motif_batch_drug1, motif_batch_idx_drug1,
                    motif_batch_drug2, motif_batch_idx_drug2
                )  # [batch, gcn_feature_dim]

                # Transformer 特征
                if drug1_indices.dim() == 0:
                    drug1_indices = drug1_indices.unsqueeze(0)
                if drug2_indices.dim() == 0:
                    drug2_indices = drug2_indices.unsqueeze(0)
                transformer_list = []
                for i in range(batch_size):
                    d1 = drug1_indices[i].item()
                    d2 = drug2_indices[i].item()
                    tf_feat = model.get_transformer_features(d1, d2)
                    transformer_list.append(tf_feat)
                transformer_features = torch.cat(transformer_list, dim=0)  # [batch, 2*d_model]

                # 细胞系投影特征
                cell_projected = model.cell_projection(cell_features)      # [batch, cell_output_dim]

                # 拼接
                combined = torch.cat([transformer_features, gcn_features, cell_projected], dim=1)
            else:
                # ===== 原始输入特征（训练前，不经过任何可学习层） =====
                # 1. 原子特征的平均池化（图级别原始特征）
                # masks: 0 表示有效节点，1 表示填充（与常见 convention 相反，但此处 masks 为 0 表示有效）
                # 需要将 masks 转换为有效节点掩码（True 表示有效）
                valid_mask = (masks == 0)  # [batch, max_atoms]
                # 对每个样本的有效节点特征求平均
                atom_features_list = []
                for i in range(batch_size):
                    valid = valid_mask[i]  # [max_atoms]
                    if valid.sum() > 0:
                        avg_feat = embeds[i, valid].mean(dim=0)  # [atom_dim]
                    else:
                        avg_feat = torch.zeros(embeds.size(-1), device=embeds.device)
                    atom_features_list.append(avg_feat)
                atom_pooled = torch.stack(atom_features_list)  # [batch, atom_dim]

                # 2. 原始药物 ECFP 特征（直接从 heterogeneous_data 中取）
                drug_features = torch.FloatTensor(heterogeneous_data['drug_features']).to(device)  # [num_drugs, drug_input_dim]
                drug1_idx = drug1_indices.long()
                drug2_idx = drug2_indices.long()
                drug1_ecfp = drug_features[drug1_idx]  # [batch, drug_input_dim]
                drug2_ecfp = drug_features[drug2_idx]  # [batch, drug_input_dim]
                # 拼接两个药物的 ECFP
                ecfp_concat = torch.cat([drug1_ecfp, drug2_ecfp], dim=1)  # [batch, 2*drug_input_dim]

                # 3. 原始细胞特征（不经过投影）
                cell_original = cell_features  # [batch, cell_feature_dim]

                # 拼接所有原始特征
                combined = torch.cat([atom_pooled, ecfp_concat, cell_original], dim=1)

            all_features.append(combined.cpu().numpy())
            all_drug1_names.extend(drug1_names)
            all_drug2_names.extend(drug2_names)
            all_cell_line_names.extend(cell_line_names)
            all_true_labels.extend(targets.cpu().numpy().flatten())

    features_array = np.vstack(all_features)
    save_dict = {
        'features': features_array,
        'drug1_names': all_drug1_names,
        'drug2_names': all_drug2_names,
        'cell_line_names': all_cell_line_names,
        'true_labels': np.array(all_true_labels)
    }
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"特征已保存到: {save_path}, 特征形状: {features_array.shape}")
# =========================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("=" * 80)
    print("整合模型训练开始")
    print("=" * 80)

    print("\n1. 加载数据...")
    data_loader = HeterogeneousDataLoader(CONFIG)
    train_loader, val_loader, test_loader, heterogeneous_data = data_loader.get_data_loaders()
    if train_loader is None:
        print("错误: 无法加载数据")
        return

    print("\n2. 创建整合模型...")
    model = IntegratedDrugSynergyModel().to(device)
    model.initialize_transformer(heterogeneous_data)
    print(f"模型参数数量: {count_parameters(model):,}")

    # ===== 训练前提取原始特征（use_learned=False）=====
    if CONFIG['logging']['save_checkpoints']:
        before_path = os.path.join(CONFIG['logging']['checkpoint_dir'], "train_features_before.pkl")
        extract_features_and_save(model, train_loader, device, before_path,
                                  heterogeneous_data=heterogeneous_data, use_learned=False)
    # =================================================

    print("\n3. 设置优化器和调度器...")
    optimizer_config = CONFIG['train']['optimizer']
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['train']['lr'],
        weight_decay=optimizer_config['weight_decay'],
        betas=tuple(optimizer_config['betas']),
        eps=optimizer_config['eps']
    )

    scheduler = None
    if CONFIG['train']['scheduler']['enabled']:
        if CONFIG['train']['scheduler']['type'] == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max' if CONFIG['early_stopping']['mode'] == 'max' else 'min',
                factor=CONFIG['train']['scheduler']['factor'],
                patience=CONFIG['train']['scheduler']['patience'],
                min_lr=CONFIG['train']['scheduler']['min_lr'],
                verbose=True
            )

    history = []
    best_val_score = -float('inf') if CONFIG['early_stopping']['mode'] == 'max' else float('inf')
    best_epoch = 0
    patience_counter = 0
    os.makedirs(CONFIG['logging']['checkpoint_dir'], exist_ok=True)

    print("\n4. 开始训练...")
    for epoch in range(CONFIG['train']['epochs']):
        start_time = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler)
        val_metrics, _, _, _ = validate(model, val_loader)

        if scheduler is not None and CONFIG['train']['scheduler']['type'] == 'ReduceLROnPlateau':
            monitor_key = CONFIG['early_stopping']['monitor'].replace('val_', '')
            scheduler.step(val_metrics[monitor_key])

        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{CONFIG['train']['epochs']} ({epoch_time:.2f}s)")
        print(f"训练 - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"验证 - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, F1: {val_metrics['f1']:.4f}")

        epoch_record = {
            'epoch': epoch+1,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['acc'],
            'train_f1': train_metrics['f1'],
            'train_auc': train_metrics.get('auc', 0),
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['acc'],
            'val_f1': val_metrics['f1'],
            'val_auc': val_metrics.get('auc', 0),
            'lr': optimizer.param_groups[0]['lr']
        }
        history.append(epoch_record)

        monitor_key = CONFIG['early_stopping']['monitor'].replace('val_', '')
        current_score = val_metrics[monitor_key]

        if (CONFIG['early_stopping']['mode'] == 'max' and current_score > best_val_score) or \
           (CONFIG['early_stopping']['mode'] == 'min' and current_score < best_val_score):
            best_val_score = current_score
            best_epoch = epoch+1
            patience_counter = 0
            if CONFIG['logging']['save_checkpoints']:
                best_path = os.path.join(CONFIG['logging']['checkpoint_dir'], "best_integrated_model.pth")
                save_model(model, best_path, epoch+1, val_metrics)
                print(f"保存最佳模型 {CONFIG['early_stopping']['monitor']}: {best_val_score:.4f}")
        else:
            patience_counter += 1
            if CONFIG['early_stopping']['enabled'] and patience_counter >= CONFIG['early_stopping']['patience']:
                print(f"\n早停触发: {patience_counter} 个epoch未改善")
                break

        if (epoch+1) % 10 == 0 and CONFIG['logging']['save_checkpoints']:
            chk_path = os.path.join(CONFIG['logging']['checkpoint_dir'], f"checkpoint_epoch_{epoch+1}.pth")
            save_model(model, chk_path, epoch+1, val_metrics)

    history_df = pd.DataFrame(history)
    history_path = os.path.join(CONFIG['logging']['checkpoint_dir'], "integrated_training_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"\n训练历史保存到: {history_path}")
    print(f"\n训练完成! 最佳模型在第 {best_epoch} 个epoch")

    print("\n5. 开始测试...")
    best_model_path = os.path.join(CONFIG['logging']['checkpoint_dir'], "best_integrated_model.pth")
    if os.path.exists(best_model_path):
        print(f"加载最佳模型: {best_model_path}")
        load_model(model, best_model_path)
    else:
        print("未找到保存的最佳模型，使用最终模型")

    # ===== 训练后提取学习后的特征（use_learned=True）=====
    if CONFIG['logging']['save_checkpoints']:
        after_path = os.path.join(CONFIG['logging']['checkpoint_dir'], "train_features_after.pkl")
        extract_features_and_save(model, train_loader, device, after_path,
                                  heterogeneous_data=heterogeneous_data, use_learned=True)
    # ====================================================

    test_metrics, test_predictions_df = validate(model, test_loader, return_predictions=True)
    test_predictions_path = os.path.join(CONFIG['logging']['checkpoint_dir'], "test_predictions.csv")
    test_predictions_df.to_csv(test_predictions_path, index=False)

    print("=" * 80)
    print("测试结果:")
    print(f"测试 Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['acc']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    if 'auc' in test_metrics:
        print(f"AUC: {test_metrics['auc']:.4f}")
    print(f"测试预测结果已保存到: {test_predictions_path}")
    print("=" * 80)

    print("\n前10个测试样本的预测结果:")
    print(test_predictions_df.head(10).to_string())

    print(f"\n训练总结:")
    print(f"- 最佳epoch: {best_epoch}")
    print(f"- 最佳验证 {CONFIG['early_stopping']['monitor'].replace('val_', '')}: {best_val_score:.4f}")
    print(f"- 测试 F1 Score: {test_metrics['f1']:.4f}")
    print(f"- 模型保存到: {CONFIG['logging']['checkpoint_dir']}")
    print(f"- 训练历史: {history_path}")
    print(f"- 测试预测结果: {test_predictions_path}")
    print(f"- 训练前特征: {os.path.join(CONFIG['logging']['checkpoint_dir'], 'train_features_before.pkl')}")
    print(f"- 训练后特征: {os.path.join(CONFIG['logging']['checkpoint_dir'], 'train_features_after.pkl')}")

    return model, history_df, test_predictions_df

if __name__ == "__main__":
    main()'''
import os
import pickle
import torch
import torch.optim as optim
import numpy as np
import time
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from config import CONFIG
from dataset import HeterogeneousDataLoader
from model import IntegratedDrugSynergyModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, path, epoch=None, metrics=None):
    state = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'metrics': metrics
    }
    torch.save(state, path)
    print(f"模型保存到: {path}")

def load_model(model, path):
    if not os.path.exists(path):
        print(f"模型文件不存在: {path}")
        return None
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    print(f"模型加载自 {path} (epoch: {state.get('epoch', 'N/A')})")
    return state.get('metrics', {})

from torch_geometric.data import Batch

def move_data_to_device(data, device):
    (embeds, adjs, masks, cnn_masks, targets,
     cell_features, drug1_nodes, drug1_indices, drug2_indices,
     drug1_names, drug2_names, cell_line_names,
     motif_batch_drug1, motif_batch_idx_drug1,
     motif_batch_drug2, motif_batch_idx_drug2) = data

    embeds = embeds.to(device)
    adjs = adjs.to(device)
    masks = masks.to(device)
    cnn_masks = cnn_masks.to(device)

    if CONFIG['data']['is_binary']:
        targets = targets.to(device).float()
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
    else:
        targets = targets.to(device).long()

    cell_features = cell_features.to(device)
    drug1_nodes = drug1_nodes.to(device)
    drug1_indices = drug1_indices.to(device)
    drug2_indices = drug2_indices.to(device)

    # 处理 motif_batch_drug1：如果是列表，先转换为 Batch 并移到设备
    if motif_batch_drug1 is not None:
        if isinstance(motif_batch_drug1, list):
            motif_batch_drug1 = [d.to(device) for d in motif_batch_drug1]
            motif_batch_drug1 = Batch.from_data_list(motif_batch_drug1)
        else:
            motif_batch_drug1 = motif_batch_drug1.to(device)
    if motif_batch_idx_drug1 is not None:
        motif_batch_idx_drug1 = motif_batch_idx_drug1.to(device)

    if motif_batch_drug2 is not None:
        if isinstance(motif_batch_drug2, list):
            motif_batch_drug2 = [d.to(device) for d in motif_batch_drug2]
            motif_batch_drug2 = Batch.from_data_list(motif_batch_drug2)
        else:
            motif_batch_drug2 = motif_batch_drug2.to(device)
    if motif_batch_idx_drug2 is not None:
        motif_batch_idx_drug2 = motif_batch_idx_drug2.to(device)

    return (embeds, adjs, masks, cnn_masks, targets,
            cell_features, drug1_nodes, drug1_indices, drug2_indices,
            drug1_names, drug2_names, cell_line_names,
            motif_batch_drug1, motif_batch_idx_drug1,
            motif_batch_drug2, motif_batch_idx_drug2)

def calculate_metrics(y_true, y_pred, y_scores=None):
    if y_true.dim() > 1:
        y_true = y_true.squeeze()
    if y_pred.dim() > 1:
        y_pred = y_pred.squeeze()

    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    metrics = {}
    try:
        metrics['acc'] = accuracy_score(y_true_np, y_pred_np)
        metrics['f1'] = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        metrics['precision'] = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        metrics['recall'] = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)

        if y_scores is not None and len(np.unique(y_true_np)) > 1:
            if CONFIG['data']['is_binary']:
                if y_scores.dim() > 1:
                    y_scores = y_scores.squeeze(1)
                y_probs = torch.sigmoid(y_scores).cpu().numpy()
                metrics['auc'] = roc_auc_score(y_true_np, y_probs)
            else:
                y_probs = torch.softmax(y_scores, dim=1).cpu().numpy()
                metrics['auc'] = roc_auc_score(y_true_np, y_probs, multi_class='ovr')
    except Exception as e:
        print(f"计算指标时出错: {e}")
        metrics = {'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0}
    return metrics

def train_epoch(model, dataloader, optimizer, scheduler=None):
    model.train()
    total_loss = 0
    all_true = []
    all_pred = []
    all_scores = []

    for batch in dataloader:
        data = move_data_to_device(batch, device)
        model_inputs = (
            data[0], data[1], data[2], data[3], data[4],
            data[5], data[6], data[7], data[8],
            data[12], data[13], data[14], data[15]
        )
        scores, loss, loss_dict = model(*model_inputs)

        optimizer.zero_grad()
        loss.backward()
        if CONFIG['train']['grad_clip']['enabled']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['train']['grad_clip']['max_norm'])
        optimizer.step()

        total_loss += loss.item()

        if CONFIG['data']['is_binary']:
            if scores.dim() > 1:
                scores_flat = scores.squeeze(1)
            else:
                scores_flat = scores
            preds = (torch.sigmoid(scores_flat) > 0.5).long()
        else:
            preds = torch.argmax(scores, dim=1)

        if preds.dim() == 0:
            preds = preds.unsqueeze(0)

        targets_for_metrics = data[4]
        if targets_for_metrics.dim() > 1:
            targets_for_metrics = targets_for_metrics.squeeze(1)

        all_true.append(targets_for_metrics.cpu())
        all_pred.append(preds.cpu())
        all_scores.append(scores.detach().cpu())

    if scheduler is not None and CONFIG['train']['scheduler']['type'] != 'ReduceLROnPlateau':
        scheduler.step()

    all_true = torch.cat(all_true)
    all_pred = torch.cat(all_pred)
    all_scores = torch.cat(all_scores)

    metrics = calculate_metrics(all_true, all_pred, all_scores)
    metrics['loss'] = total_loss / len(dataloader)
    return metrics

def validate(model, dataloader, return_predictions=False):
    model.eval()
    total_loss = 0
    all_true = []
    all_pred = []
    all_scores = []
    all_drug1_names = []
    all_drug2_names = []
    all_cell_line_names = []

    with torch.no_grad():
        for batch in dataloader:
            data = move_data_to_device(batch, device)
            model_inputs = (
                data[0], data[1], data[2], data[3], data[4],
                data[5], data[6], data[7], data[8],
                data[12], data[13], data[14], data[15]
            )
            scores, loss, loss_dict = model(*model_inputs)

            total_loss += loss.item()

            if CONFIG['data']['is_binary']:
                if scores.dim() > 1:
                    scores_flat = scores.squeeze(1)
                else:
                    scores_flat = scores
                preds = (torch.sigmoid(scores_flat) > 0.5).long()
            else:
                preds = torch.argmax(scores, dim=1)

            if preds.dim() == 0:
                preds = preds.unsqueeze(0)

            targets_for_metrics = data[4]
            if targets_for_metrics.dim() > 1:
                targets_for_metrics = targets_for_metrics.squeeze(1)

            all_true.append(targets_for_metrics.cpu())
            all_pred.append(preds.cpu())
            all_scores.append(scores.cpu())

            all_drug1_names.extend(data[9])
            all_drug2_names.extend(data[10])
            all_cell_line_names.extend(data[11])

    all_true = torch.cat(all_true)
    all_pred = torch.cat(all_pred)
    all_scores = torch.cat(all_scores)
    metrics = calculate_metrics(all_true, all_pred, all_scores)
    metrics['loss'] = total_loss / len(dataloader)

    if return_predictions:
        if CONFIG['data']['is_binary']:
            scores_np = all_scores.cpu().numpy()
            if scores_np.ndim > 1:
                scores_np = scores_np.squeeze(1)
            probs_np = torch.sigmoid(all_scores).cpu().numpy()
            if probs_np.ndim > 1:
                probs_np = probs_np.squeeze(1)
        else:
            scores_np = all_scores.cpu().numpy().tolist()
            probs_np = torch.softmax(all_scores, dim=1).cpu().numpy().tolist()

        predictions_df = pd.DataFrame({
            'drug1': all_drug1_names,
            'drug2': all_drug2_names,
            'cell_line': all_cell_line_names,
            'true_label': all_true.cpu().numpy(),
            'predicted_label': all_pred.cpu().numpy(),
            'prediction_score': scores_np,
            'predicted_probability': probs_np
        })
        return metrics, predictions_df
    else:
        return metrics, all_true, all_pred, all_scores

# ========== 特征提取函数（修改为支持返回数据而不保存） ==========
def extract_features(model, dataloader, device, heterogeneous_data=None, use_learned=True):
    """
    提取特征，返回特征字典，不保存文件。
    Args:
        model: 模型实例
        dataloader: 数据加载器
        device: 设备
        heterogeneous_data: 异构网络数据（包含 drug_features 等），当 use_learned=False 时需要
        use_learned: True 表示提取经过模型学习后的特征；False 表示提取原始输入特征
    Returns:
        字典包含 features, drug1_names, drug2_names, cell_line_names, true_labels
    """
    model.eval()
    all_features = []
    all_drug1_names = []
    all_drug2_names = []
    all_cell_line_names = []
    all_true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            data = move_data_to_device(batch, device)
            embeds, adjs, masks, cnn_masks, targets, cell_features, drug1_nodes, drug1_indices, drug2_indices = data[:9]
            drug1_names, drug2_names, cell_line_names = data[9], data[10], data[11]
            motif_batch_drug1, motif_batch_idx_drug1, motif_batch_drug2, motif_batch_idx_drug2 = data[12:16]

            batch_size = embeds.size(0)

            if use_learned:
                # ===== 学习后的特征 =====
                _, _, gcn_features = model.gcn_model(
                    embeds, adjs, masks, cnn_masks, targets,
                    cell_features, drug1_nodes,
                    motif_batch_drug1, motif_batch_idx_drug1,
                    motif_batch_drug2, motif_batch_idx_drug2
                )

                if drug1_indices.dim() == 0:
                    drug1_indices = drug1_indices.unsqueeze(0)
                if drug2_indices.dim() == 0:
                    drug2_indices = drug2_indices.unsqueeze(0)
                transformer_list = []
                for i in range(batch_size):
                    d1 = drug1_indices[i].item()
                    d2 = drug2_indices[i].item()
                    tf_feat = model.get_transformer_features(d1, d2)
                    transformer_list.append(tf_feat)
                transformer_features = torch.cat(transformer_list, dim=0)

                cell_projected = model.cell_projection(cell_features)

                combined = torch.cat([transformer_features, gcn_features, cell_projected], dim=1)
            else:
                # ===== 原始输入特征 =====
                valid_mask = (masks == 0)
                atom_features_list = []
                for i in range(batch_size):
                    valid = valid_mask[i]
                    if valid.sum() > 0:
                        avg_feat = embeds[i, valid].mean(dim=0)
                    else:
                        avg_feat = torch.zeros(embeds.size(-1), device=embeds.device)
                    atom_features_list.append(avg_feat)
                atom_pooled = torch.stack(atom_features_list)

                drug_features = torch.FloatTensor(heterogeneous_data['drug_features']).to(device)
                drug1_idx = drug1_indices.long()
                drug2_idx = drug2_indices.long()
                drug1_ecfp = drug_features[drug1_idx]
                drug2_ecfp = drug_features[drug2_idx]
                ecfp_concat = torch.cat([drug1_ecfp, drug2_ecfp], dim=1)

                cell_original = cell_features

                combined = torch.cat([atom_pooled, ecfp_concat, cell_original], dim=1)

            all_features.append(combined.cpu().numpy())
            all_drug1_names.extend(drug1_names)
            all_drug2_names.extend(drug2_names)
            all_cell_line_names.extend(cell_line_names)
            all_true_labels.extend(targets.cpu().numpy().flatten())

    features_array = np.vstack(all_features)
    return {
        'features': features_array,
        'drug1_names': all_drug1_names,
        'drug2_names': all_drug2_names,
        'cell_line_names': all_cell_line_names,
        'true_labels': np.array(all_true_labels)
    }
# =========================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("=" * 80)
    print("整合模型训练开始")
    print("=" * 80)

    print("\n1. 加载数据...")
    data_loader = HeterogeneousDataLoader(CONFIG)
    train_loader, val_loader, test_loader, heterogeneous_data = data_loader.get_data_loaders()
    if train_loader is None:
        print("错误: 无法加载数据")
        return

    print("\n2. 创建整合模型...")
    model = IntegratedDrugSynergyModel().to(device)
    model.initialize_transformer(heterogeneous_data)
    print(f"模型参数数量: {count_parameters(model):,}")

    # ===== 训练前提取原始特征，合并保存为一个文件 =====
    if CONFIG['logging']['save_checkpoints']:
        print("\n提取训练前特征...")
        before_data = {
            'features': [],
            'drug1_names': [],
            'drug2_names': [],
            'cell_line_names': [],
            'true_labels': [],
            'dataset_type': []  # 新增字段标识样本来自哪个数据集
        }

        for name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
            print(f"  处理 {name} 集...")
            feat_dict = extract_features(model, loader, device,
                                         heterogeneous_data=heterogeneous_data,
                                         use_learned=False)
            before_data['features'].append(feat_dict['features'])
            before_data['drug1_names'].extend(feat_dict['drug1_names'])
            before_data['drug2_names'].extend(feat_dict['drug2_names'])
            before_data['cell_line_names'].extend(feat_dict['cell_line_names'])
            before_data['true_labels'].append(feat_dict['true_labels'])
            before_data['dataset_type'].extend([name] * len(feat_dict['true_labels']))

        # 合并特征和标签
        before_data['features'] = np.vstack(before_data['features'])
        before_data['true_labels'] = np.concatenate(before_data['true_labels'])
        before_data['dataset_type'] = np.array(before_data['dataset_type'])

        before_path = os.path.join(CONFIG['logging']['checkpoint_dir'], "all_features_before.pkl")
        with open(before_path, 'wb') as f:
            pickle.dump(before_data, f)
        print(f"训练前特征已保存到: {before_path}, 总样本数: {len(before_data['true_labels'])}")

    print("\n3. 设置优化器和调度器...")
    optimizer_config = CONFIG['train']['optimizer']
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['train']['lr'],
        weight_decay=optimizer_config['weight_decay'],
        betas=tuple(optimizer_config['betas']),
        eps=optimizer_config['eps']
    )

    scheduler = None
    if CONFIG['train']['scheduler']['enabled']:
        if CONFIG['train']['scheduler']['type'] == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max' if CONFIG['early_stopping']['mode'] == 'max' else 'min',
                factor=CONFIG['train']['scheduler']['factor'],
                patience=CONFIG['train']['scheduler']['patience'],
                min_lr=CONFIG['train']['scheduler']['min_lr'],
                verbose=True
            )

    history = []
    best_val_score = -float('inf') if CONFIG['early_stopping']['mode'] == 'max' else float('inf')
    best_epoch = 0
    patience_counter = 0
    os.makedirs(CONFIG['logging']['checkpoint_dir'], exist_ok=True)

    print("\n4. 开始训练...")
    for epoch in range(CONFIG['train']['epochs']):
        start_time = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler)
        val_metrics, _, _, _ = validate(model, val_loader)

        if scheduler is not None and CONFIG['train']['scheduler']['type'] == 'ReduceLROnPlateau':
            monitor_key = CONFIG['early_stopping']['monitor'].replace('val_', '')
            scheduler.step(val_metrics[monitor_key])

        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{CONFIG['train']['epochs']} ({epoch_time:.2f}s)")
        print(f"训练 - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"验证 - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, F1: {val_metrics['f1']:.4f}")

        epoch_record = {
            'epoch': epoch+1,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['acc'],
            'train_f1': train_metrics['f1'],
            'train_auc': train_metrics.get('auc', 0),
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['acc'],
            'val_f1': val_metrics['f1'],
            'val_auc': val_metrics.get('auc', 0),
            'lr': optimizer.param_groups[0]['lr']
        }
        history.append(epoch_record)

        monitor_key = CONFIG['early_stopping']['monitor'].replace('val_', '')
        current_score = val_metrics[monitor_key]

        if (CONFIG['early_stopping']['mode'] == 'max' and current_score > best_val_score) or \
           (CONFIG['early_stopping']['mode'] == 'min' and current_score < best_val_score):
            best_val_score = current_score
            best_epoch = epoch+1
            patience_counter = 0
            if CONFIG['logging']['save_checkpoints']:
                best_path = os.path.join(CONFIG['logging']['checkpoint_dir'], "best_integrated_model.pth")
                save_model(model, best_path, epoch+1, val_metrics)
                print(f"保存最佳模型 {CONFIG['early_stopping']['monitor']}: {best_val_score:.4f}")
        else:
            patience_counter += 1
            if CONFIG['early_stopping']['enabled'] and patience_counter >= CONFIG['early_stopping']['patience']:
                print(f"\n早停触发: {patience_counter} 个epoch未改善")
                break

        if (epoch+1) % 10 == 0 and CONFIG['logging']['save_checkpoints']:
            chk_path = os.path.join(CONFIG['logging']['checkpoint_dir'], f"checkpoint_epoch_{epoch+1}.pth")
            save_model(model, chk_path, epoch+1, val_metrics)

    history_df = pd.DataFrame(history)
    history_path = os.path.join(CONFIG['logging']['checkpoint_dir'], "integrated_training_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"\n训练历史保存到: {history_path}")
    print(f"\n训练完成! 最佳模型在第 {best_epoch} 个epoch")

    print("\n5. 开始测试...")
    best_model_path = os.path.join(CONFIG['logging']['checkpoint_dir'], "best_integrated_model.pth")
    if os.path.exists(best_model_path):
        print(f"加载最佳模型: {best_model_path}")
        load_model(model, best_model_path)
    else:
        print("未找到保存的最佳模型，使用最终模型")

    # ===== 训练后提取学习后特征，合并保存为一个文件 =====
    if CONFIG['logging']['save_checkpoints']:
        print("\n提取训练后特征...")
        after_data = {
            'features': [],
            'drug1_names': [],
            'drug2_names': [],
            'cell_line_names': [],
            'true_labels': [],
            'dataset_type': []
        }

        for name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
            print(f"  处理 {name} 集...")
            feat_dict = extract_features(model, loader, device,
                                         heterogeneous_data=heterogeneous_data,
                                         use_learned=True)
            after_data['features'].append(feat_dict['features'])
            after_data['drug1_names'].extend(feat_dict['drug1_names'])
            after_data['drug2_names'].extend(feat_dict['drug2_names'])
            after_data['cell_line_names'].extend(feat_dict['cell_line_names'])
            after_data['true_labels'].append(feat_dict['true_labels'])
            after_data['dataset_type'].extend([name] * len(feat_dict['true_labels']))

        after_data['features'] = np.vstack(after_data['features'])
        after_data['true_labels'] = np.concatenate(after_data['true_labels'])
        after_data['dataset_type'] = np.array(after_data['dataset_type'])

        after_path = os.path.join(CONFIG['logging']['checkpoint_dir'], "all_features_after.pkl")
        with open(after_path, 'wb') as f:
            pickle.dump(after_data, f)
        print(f"训练后特征已保存到: {after_path}, 总样本数: {len(after_data['true_labels'])}")

    test_metrics, test_predictions_df = validate(model, test_loader, return_predictions=True)
    test_predictions_path = os.path.join(CONFIG['logging']['checkpoint_dir'], "test_predictions.csv")
    test_predictions_df.to_csv(test_predictions_path, index=False)

    print("=" * 80)
    print("测试结果:")
    print(f"测试 Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['acc']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    if 'auc' in test_metrics:
        print(f"AUC: {test_metrics['auc']:.4f}")
    print(f"测试预测结果已保存到: {test_predictions_path}")
    print("=" * 80)

    print("\n前10个测试样本的预测结果:")
    print(test_predictions_df.head(10).to_string())

    print(f"\n训练总结:")
    print(f"- 最佳epoch: {best_epoch}")
    print(f"- 最佳验证 {CONFIG['early_stopping']['monitor'].replace('val_', '')}: {best_val_score:.4f}")
    print(f"- 测试 F1 Score: {test_metrics['f1']:.4f}")
    print(f"- 模型保存到: {CONFIG['logging']['checkpoint_dir']}")
    print(f"- 训练历史: {history_path}")
    print(f"- 测试预测结果: {test_predictions_path}")
    print(f"- 训练前特征: {os.path.join(CONFIG['logging']['checkpoint_dir'], 'all_features_before.pkl')}")
    print(f"- 训练后特征: {os.path.join(CONFIG['logging']['checkpoint_dir'], 'all_features_after.pkl')}")

    return model, history_df, test_predictions_df

if __name__ == "__main__":
    main()