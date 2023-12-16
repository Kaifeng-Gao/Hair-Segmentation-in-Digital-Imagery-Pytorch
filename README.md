# Hair-Segmentation-in-Digital-Imagery-Pytorch

Pretrained DeepLabv3+, PAN, PSPNet for Figaro1k, implemented in Pytorch.

## Quick Start
1. Prepare the environment: `pip install -r requirements.txt`
2. Download pretrained model ([Google Drive][1]) and put it under `./checkpoints`:
3. (Optional) Download dataset ([Figaro1K][2]) and put it under `./datasets/data`: 
4. Load pretrained model and visualize segmentation result on test set
```bash
python main.py --test_only --save_val_results --download --ckpt checkpoints/best_PAN_resnet101_hair_os16.pth --model PAN
```

![][3]
## Prediction
1. Download pretrained model ([Google Drive][4]) and put it under `./checkpoints`:
2. Load pretrained model and predict any images:
```bash
python predict.py --input path_to_img --dataset hair --ckpt checkpoints/best_PAN_resnet101_hair_os16.pth --model PAN --save_val_results_to results 
```

## Training

### Dataset Options

| Parameter     | Default             | Options  | Description          |
| ------------- | ------------------- | -------- | -------------------- |
| `--data_root` | `'./datasets/data'` |          | Path to the dataset. |
| `--dataset`   | `'hair'`            | `'hair'` | Name of the dataset. |

### Model Options

| Parameter         | Default           | Options                                                        | Description                     |
| ----------------- | ----------------- | -------------------------------------------------------------- | ------------------------------- |
| `--model`         | `'DeepLabV3Plus'` | Unet, UnetPlusPlus, FPN, PAN, PSPNet, DeepLabV3, DeepLabV3Plus | Model architecture.             |
| `--encoder`       | `'resnet101'`     | mobilenet\_v2, resnet101, resnet50                             | Encoder/Backbone for the model. |
| `--output_stride` | `16`              | `8`, `16`                                                      | Output stride for the model.    |

### Train Options

| Parameter             | Default           | Options                           | Description                                     |
| --------------------- | ----------------- | --------------------------------- | ----------------------------------------------- |
| `--test_only`         | `False`           |                                   | Test the model.                                 |
| `--save_val_results`  | `False`           |                                   | Save validation results to `"./results"`.       |
| `--total_itrs`        | `21000`           |                                   | Total number of training iterations.            |
| `--lr`                | `0.002`           |                                   | Learning rate.                                  |
| `--lr_policy`         | `'poly'`          | `'poly'`, `'step'`                | Learning rate scheduler policy.                 |
| `--step_size`         | `10000`           |                                   | Step size for the step learning rate scheduler. |
| `--batch_size`        | `4`               |                                   | Batch size for training.                        |
| `--val_batch_size`    | `1`               |                                   | Batch size for validation.                      |
| `--crop_size`         | `256`             |                                   | Crop size for training images.                  |
| `--ckpt`              | `None`            |                                   | Restore from checkpoint.                        |
| `--continue_training` | `False`           |                                   | Continue training from checkpoint.              |
| `--loss_type`         | `'cross_entropy'` | `'cross_entropy'`, `'focal_loss'` | Type of loss function.                          |
| `--gpu_id`            | `'0'`             |                                   | GPU ID to use.                                  |
| `--weight_decay`      | `1e-4`            |                                   | Weight decay for optimization.                  |
| `--random_seed`       | `1`               |                                   | Random seed for initialization.                 |
| `--val_interval`      | `210`             |                                   | Epoch interval for evaluation.                  |
| `--download`          | `True`            |                                   | Automatically download the dataset.             |

```bash
python main.py --dataset hair --download --model PAN --total_itrs 100 --val_interval 10
```

## Results

| Parameter      | Value   |
|----------------|---------|
| Epoch Number   | 100     |
| Learning Rate  | 2e-3    |
| Batch Size     | 4       |
| Weight Decay   | 1e-4    |
| Momentum       | 0.9     |
| Encoder        | ResNet101    |
| Optimizer      | SGD          |
| Loss Function  | Cross Entropy|

| Model         | Overall Accuracy | Mean IoU | F1-Score |
| :------------ | :--------------: | :------: | :------: |
| PSPNet        | 0.940            | 0.885    | 0.939    |
| DeepLabV3Plus | 0.958            | 0.919    | 0.958    |
| PAN           | 0.963            | 0.929    | 0.963    |


[1]:	https://drive.google.com/drive/folders/189EuKz8rBfSMRLZyJnkAJRuLvMonAHgG?usp=sharing
[2]:	http://projects.i-ctm.eu/it/progetto/figaro-1k
[3]:	https://windypic.oss-cn-guangzhou.aliyuncs.com/image/CleanShot%202023-12-15%20at%2021.01.11@2x.png
[4]:	https://drive.google.com/drive/folders/189EuKz8rBfSMRLZyJnkAJRuLvMonAHgG?usp=sharing