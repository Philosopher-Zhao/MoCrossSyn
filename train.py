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
from model import MoCrossSyn
from torch_geometric.data import Batch

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

    # 处理 motif_batch_drug1
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
    model = MoCrossSyn().to(device)
    model.initialize_transformer(heterogeneous_data)
    print(f"模型参数数量: {count_parameters(model):,}")

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

    return model, history_df, test_predictions_df

if __name__ == "__main__":
    main()
