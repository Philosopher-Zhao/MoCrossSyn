import os
import yaml

# 全局配置变量
_CONFIG = None


def load_integrated_config(config_path='config_integrated.yml'):
    """加载并解析整合模型的配置文件"""
    global _CONFIG

    if not os.path.exists(config_path):
        print(f"警告: 配置文件 {config_path} 不存在，使用默认配置")
        # 创建默认配置
        config = get_default_config()
        _CONFIG = config
        return config

    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        # 自动根据num_classes配置模式
        if config['data']['num_classes'] == 1:
            config['data']['actual_output_dim'] = 1
            config['data']['is_binary'] = True
        else:
            config['data']['actual_output_dim'] = config['data']['num_classes']
            config['data']['is_binary'] = False

        # 确保必要的目录存在
        os.makedirs(config['logging']['log_dir'], exist_ok=True)
        os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['paths']['output_dir'], exist_ok=True)

        _CONFIG = config
        return config

    except Exception as e:
        print(f"加载配置文件失败: {e}")
        print("使用默认配置")
        config = get_default_config()
        _CONFIG = config
        return config


def get_default_config():
    """返回默认配置"""
    return {
        'device': 'cuda',
        'paths': {
            'drug_smiles': '../Dataset/drug_smiles.csv',
            'drug_combinations': '../Dataset/drug_drug_cell.csv',
            'cell_features': '../Dataset/cell_features.csv',
            'output_dir': 'output',
            'drug_target_interactions': '../Dataset/drug_targets.csv',
            'target_target_interactions': '../Dataset/target_interactions.csv',
            'drug_drug_interactions': '../Dataset/drug_interactions.csv',
            'target_features': '../Dataset/target_features.csv'
        },
        'transformer': {
            'drug_input_dim': 1024,
            'target_input_dim': 1024,
            'd_model': 256,
            'num_heads': 8,
            'dff': 512,
            'num_layers': 4,
            'dropout_rate': 0.1
        },
        'data': {
            'atom_dim': 78,
            'atom_hid_dim': 128,
            'max_atoms': 100,
            'cell_feature_dim': 1000,
            'num_classes': 1,
            'actual_output_dim': 1,
            'is_binary': True
        },
        'model': {
            'block': {
                'num': 4,
                'ffn': {'hidden_dim': 256}
            },
            'cross_attn_heads': 8,
            'attn_heads': 8,
            'is_cat_after_readout': True,
            'is_rg': True,
            'is_gcn': False,
            'is_joint': True,
            'is_alpha_learn': True,
            'is_init_gru': False,
            'alpha': 0.5,
            'is_a_': True,
            'cell_projection': {'hidden_dims': [512, 256]},
            'mlp': {'hidden_dim': 128}
        },
        'train': {
            'batch_size': 32,
            'epochs': 100,
            'lr': 0.001,
            'dropout': 0.3,
            'num_workers': 4,
            'optimizer': {
                'weight_decay': 0.0001,
                'betas': [0.9, 0.999],
                'eps': 1e-08
            },
            'scheduler': {
                'enabled': True,
                'type': 'ReduceLROnPlateau',
                'factor': 0.5,
                'patience': 10,
                'min_lr': 1e-6
            },
            'grad_clip': {
                'enabled': True,
                'max_norm': 1.0
            }
        },
        'early_stopping': {
            'enabled': True,
            'patience': 20,
            'monitor': 'val_f1',
            'mode': 'max'
        },
        'logging': {
            'log_dir': 'logs',
            'checkpoint_dir': 'checkpoints',
            'save_checkpoints': True
        }
    }


def get_config():
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_integrated_config()
    return _CONFIG


CONFIG = get_config()