# MoCrossSyn

## Welcome to MoCrossSyn
**MoCrossSyn (Prediction of Drug Synergistic Combination Based on Motif Structure Recognition and Cross-Drug Interaction Modeling)** <p align="justify">  Drug combination therapy holds significant clinical value in the treatment of complex diseases such as cancer. However, given the vast number of potential drug combinations, relying solely on experimental screening is both time-consuming and costly; therefore, the development of efficient computational prediction methods is of great importance. In recent years, deep learning models have made significant progress in drug synergy prediction tasks. However, existing methods still face challenges in simultaneously capturing molecular structural information and complex biological network relationships. To address these, we propose a novel drug synergy prediction model, MoCrossSyn. This model integrates multimodal information to jointly model potential synergistic relationships between drugs at both the molecular structural and systems biology levels. First, at the molecular structural level, a joint molecular graph of drug pairs is constructed, and a graph attention network is employed to model atomic-level structural information. Concurrently, a Motif structural encoding mechanism is introduced to extract key substructural fragments via BRICS molecular decomposition, thereby enhancing the model’s ability to represent functional groups and local chemical patterns. Additionally, a cross-drug attention mechanism is utilized to capture potential atomic-level interactions between different drugs. Second, at the systems biology level, we construct a heterogeneous biological network incorporating drug-target relationships and employ a Transformer-based encoder to learn the embedding representations of drugs within the biological network, thereby capturing global association information between drugs. Additionally, we use deep autoencoders to perform dimensionality reduction on cell line gene expression data to obtain cell line feature representations. Finally, molecular structural features, network embedding features, and cell line expression features are fused, and a classifier is employed to predict drug synergistic effects. Experimental results on standard drug synergy datasets demonstrate that the proposed MoCrossSyn model outperforms various baseline methods across multiple evaluation metrics, including ACC, F1-score and AUROC. Further ablation experiments indicate that both motif structural modeling and heterogeneous network representation learning significantly contribute to the model’s performance improvement. These results demonstrate that the model can effectively integrate multi-source bioinformatics data, providing an effective method for the computational prediction of drug synergistic combinations.</p>

The flow chart of MoCrossSyn is as follows:

![示例图片](./框架图.png)

## Directory Structure

```markdown
├── Datasets
│   ├── features
│   │   ├── cell_features.csv           
│   │   ├── drug_target.csv            
│   │   ├── target_features.csv
│   │   ├── target_interactions.csv
│   │   ├── drug_interactions.csv    
│   │   └── drug_smiles.csv             
│   └── samples					  
│       └── drug_drug_cell.csv            
├── config_intergrated.yml                     
├── config.py                      
├── data_preprocess.py                         
├── dateset.py                       
├── main.py                      
├── model.py                                            
└── train.py              
```

## Installation and Requirements

MoCrossSyn has been tested in a Python 3.9 environment. It is recommended to use the same library versions as specified. Using a conda virtual environment is also recommended to avoid affecting other parts of the system. Please follow the steps below.

Key libraries and versions:

```markdown
├── torch              1.12.0+cu113
├── torch-geometric    2.5.3
├── torch-scatter      2.1.2
├── torch-sparse       0.6.18
├── numpy              1.26.4
├── pandas             1.2.4
├── scikit-learn       1.4.2
├── rdkit              2024.9.5
├── pyyaml             6.0.2
└── scipy              1.10.1     
```

### Step 1: Download Code and Data

Use the following command to download this project or download the zip file from the "Code" section at the top right:

```bash
git init 
git clone https://github.com/Philosopher-Zhao/MoCrossSyn.git
```

### Step 2: Run the Model

Run the main script in the virtual environment:

```bash
python main.py
```

All results of the operation will be saved in the current directory.

## Citation 

If you use our tool and code, please cite our article and mark the project to show your support，thank you!

Citation format: 


Paper Link:
