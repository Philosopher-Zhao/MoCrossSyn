import sys
import os
# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
def check_dependencies():
    """检查依赖包"""
    required_packages = [
        'torch',
        'numpy',
        'pandas',
        'sklearn',
        'rdkit',
        'yaml'
    ]
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            elif package == 'yaml':
                __import__('yaml')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)

    return missing_packages


def main():
    """主函数"""
    print("=" * 80)
    print("整合药物协同预测模型训练开始")
    print("=" * 80)

    # 检查依赖
    print("\n检查依赖包...")
    missing_packages = check_dependencies()
    if missing_packages:
        print(f"缺少依赖包: {missing_packages}")
        print("请使用以下命令安装:")
        print("pip install torch numpy pandas scikit-learn rdkit pyyaml")
        return

    print("所有依赖包已安装!")

    # 检查配置文件
    print("\n检查配置文件...")
    config_path = './config_integrated.yml'
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在")
        print("将使用默认配置运行...")
    else:
        print(f"找到配置文件: {config_path}")

    # 检查数据文件
    print("\n检查数据文件...")
    from config import CONFIG

    required_data_files = [
        CONFIG['paths']['drug_smiles'],
        CONFIG['paths']['drug_combinations'],
        CONFIG['paths']['cell_features'],
        CONFIG['paths']['drug_target_interactions'],
        CONFIG['paths']['target_target_interactions'],
        CONFIG['paths']['drug_drug_interactions'],
        CONFIG['paths']['target_features']
    ]

    missing_files = []
    for file_path in required_data_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("以下数据文件不存在:")
        for f in missing_files:
            print(f"  - {f}")
        print("\n请确保所有数据文件都在正确的位置")
        return

    print("所有数据文件都存在!")

    # 开始训练
    print("\n开始训练整合模型...")
    try:
        from train import main as train_main
        train_main()
        print("\n整合模型训练完成!")
    except ImportError as e:
        print(f"导入错误: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
