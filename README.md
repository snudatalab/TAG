# Accurate graph classification via two-staged contrastive curriculum learning

This repository is a Pytorch implementation of *Accurate graph classification via two-staged contrastive curriculum learning*, submitted to PLOS ONE 2023.

## Settings

We implement codes based on the following environments:

- Python 3.7
- PyTorch 1.4.0
- PyTorch Geometric 1.6.3

## Datasets

We use seven datasets for graph classification.
All datasets can be downloaded from [link](https://chrsmrrs.github.io/datasets/docs/datasets/).
If you run the code, the datasets will be downloaded in the `dataset/` folder.
The detailed information on datasets is summarized in the table below.

|Name     |Graphs |Nodes  |Edges  |Features |Classes|
|:--------|------:|------:|------:|--------:|------:|
|MUTAG    |188    |3371   |3721   |7        |2      |
|PROTEINS |1113   |43471  |81044  |3        |2      |
|NCI1     |4110   |122747 |132753 |37       |2      |
|NCI109   |4127   |122494 |132604 |38       |2      |
|DD       |1178   |334925 |843046 |89       |2      |
|PTC_MR   |344    |4915   |5054   |18       |2      |
|DBLP     |19456  |203954 |764512 |41325    |2      |

## Code Information

We explain codes for **TAG** (Two-staged contrAstive curriculum learning for Graphs) and describe how to run the code.
`main.py` gets hyperparameters and conducts the overall process of TAG.
The proposed method is implemented in `model/tag.py` and proposed augmentation algorithms are implemented inside the `augment` folder.

To run this project, you have to type the following command.
```bash
python main.py
```
This command runs the code in the default setting.
