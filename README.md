# Scale-Aware Contrastive Reverse Distillation for Unsupervised Medical Anomaly Detection

This repository contains the official implementation of **Scale-Aware Contrastive Reverse Distillation for Unsupervised Medical Anomaly Detection**, accepted by ICLR 2025.

<p align="center">
  <img src="./architecture.png" width="900" alt="Network architecture of SCRD4AD">
</p>


## Environment

The code was tested with the following package versions:

```text
torch == 1.9.1
torchvision == 0.10.1
numpy == 1.20.3
scipy == 1.7.1
scikit-learn == 1.0
Pillow == 8.3.2
```

You can create the environment with:

```bash
conda create -n scrd4ad python=3.8
conda activate scrd4ad

# Please choose the CUDA version according to your machine.
conda install pytorch==1.9.1 torchvision==0.10.1 cudatoolkit=11.1 -c pytorch -c conda-forge

pip install numpy==1.20.3 scipy==1.7.1 scikit-learn==1.0 Pillow==8.3.2
```

## Dataset

Please organize the dataset as follows:

```text
datasets/
└── <dataset_name>/
    ├── train/
    ├── test/
    └── ground_truth/
```

Update the dataset path in the configuration file before training and testing.

## Training

```bash
python train.py --dataset <dataset_name> --data_path ./datasets/<dataset_name>
```

## Evaluation

```bash
python test.py --dataset <dataset_name> --data_path ./datasets/<dataset_name> --checkpoint ./checkpoints/<checkpoint_name>.pth
```

> Note: Please modify the script names and arguments according to the actual code in this repository.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{li2025scale,
  title={Scale-aware contrastive reverse distillation for unsupervised medical anomaly detection},
  author={Li, Chunlei and Shi, Yilei and Hu, Jingliang and Zhu, Xiaoxiang and Mou, Lichao},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```

## License

This project is released for academic research use. Please refer to the license file for more details.
