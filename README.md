# WQCount: Single Domain Generalization for Crowd Counting Based on Feature Decoupling
This is an official repository for our work, "WQCount: Single Domain Generalization for Crowd Counting Based on Feature Decoupling"

## Requirements
* Python 3.10.12
* PyTorch 2.0.1
* Torchvision 0.15.2
* Others specified in [requirements.txt](requirements.txt)

## Data Preparation
1. Download ShanghaiTech and UCF-QNRF datasets from official sites and unzip them.
2. Run the following commands to preprocess the datasets:
    ```
    python utils/preprocess_data.py --origin-dir [path_to_ShanghaiTech]/part_A --data-dir data/sta
    python utils/preprocess_data.py --origin-dir [path_to_ShanghaiTech]/part_B --data-dir data/stb
    python utils/preprocess_data.py --origin-dir [path_to_UCF-QNRF] --data-dir data/qnrf
    ```
3. Run the following commands to generate GT density maps:
    ```
    python dmap_gen.py --path data/sta
    python dmap_gen.py --path data/stb
    python dmap_gen.py --path data/qnrf
    ```


## Testing
Run the following commands after you specify the path to the model weight in the config file:
```
python main.py --task test --config configs/sta_test_stb.yml
python main.py --task test --config configs/sta_test_qnrf.yml
```

## Inference
Run the following command:
```
python inference.py --img_path [path_to_img_file_or_directory] --model_path [path_to_model_weight] --save_path output.txt --vis_dir vis
```

## Pretrained Weights
We provide pretrained weights in the table below:
| Source | Performance                                   | Weights                                                                                                                                          |
| ------ | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| A      | B: 9.5MAE, 16.1MSE<br>Q: 113.7MAE, 194.8MSE  | [Google Drive](https://drive.google.com/file/d/1-px7JTChkHm7OI1rcfs-wesctVHrHxz1/view?usp=sharing)|
| B      | A: 94.1MAE, 179.9MSE<br>Q: 150.7MAE, 271.3MSE | [Google Drive](https://drive.google.com/file/d/1hegMnRyI7EepwbwW-5qG7R4mlHtXcNdk/view?usp=sharing)|

[//]: # (| Q      | A: 64.9MAE, 111.0MSE<br>B: 14.3MAE, 28.7MSE   | [Download]&#40;&#41; |)

