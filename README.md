# Code for Prompting Continual Person Search

## Prerequest

1. Main dependecies

    ```shell
        torch==1.8.2
        torchvision==0.9.2
        cuda==11.1
        detectron2==0.5
        numpy=1.20.2
        opencv-python==4.8.0
    ```

2. Preparing data

    * Download [COCO](https://cocodataset.org/#download) and [CrowdHuman](https://www.crowdhuman.org/) datasets for detection pretraining.

    * Download [CUHK-SYSU](https://github.com/ShuangLI59/person_search), [PRW](https://github.com/liangzheng06/PRW-baseline) and [MovieNet-PS](https://github.com/ZhengPeng7/GLCNet) for continual person search.

    * Create folder ```Data/``` and place the data as

        ```shell
                Data
                ├── cuhk_sysu
                │   ├── annotation
                │   ├── Image
                │   └── README.txt
                ├── DetData
                │   ├── coco
                │   │   ├── annotations
                │   │   ├── train2017
                │   │   └── ......
                │   └── crowd_human
                │       ├── annotations
                │       ├── CrowdHuman_train
                │       └── ......
                ├── movienet
                │   ├── anno_mvn-cs
                │   ├── clean.py
                │   └── Image
                └── PRW
                ├── annotations
                ├── frames
                └── ......
        ```

    * Convert the annotation of CrowdHuman to the COCO format

        ```shell
        python tools/convert_crowdhuman_to_coco.py
        ```

3. Preparing model weight

    * Download pre-trained [Swin-S](https://github.com/microsoft/Swin-Transformer) weight ```swin_small_patch4_window7_224_22k.pth``` into ```Data/model_zoo/```
  
    * Convert the weight by

        ```shell
        python tools/convert_swin_weight.py Data/model_zoo/swin_small_patch4_window7_224_22k.pth  Data/model_zoo/swin_small_patch4_window7_224_22k_d2.pkl
        ```

## Training

* Train any model with the config

    ```shell
    python tools/train_ps_net.py --config-file ${file_path}  --num-gpus ${num_gpu} --resume --dist-url tcp://127.0.0.1:60888
    ```

* Run ```train.sh``` to conduct the integral training procedure

* We recommend to use [Tensorboard](https://www.tensorflow.org/tensorboard) to monitor the training process:

    ```shell
    tensorboard --logdir outputs/${model_output}$
    ```

## Inference

* Test any model with the config

    ```shell
    python tools/train_ps_net.py --config-file ${file_path}  --num-gpus ${num_gpu} --resume --dist-url tcp://127.0.0.1:60888 --eval-only
    ```

* Run ```test.sh``` after ```train.sh``` to evaluate the continual person search performance

## Code structure

* The overall structure of the code is based on [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html). This repository is organized as

    ```shell
    PoPS
    ├── configs # Where model configuration files saved
    ├── Data # Where data and model weights saved 
    ├── LICENSE
    ├── outputs # Where training logs and checkpoints saved
    ├── pops # The main code
    │   ├── checkpoint # Automatic checkpoint tool
    │   ├── config # Definition of the basic configuration for all models
    │   ├── data # Reading and augmenting data
    │   ├── engine # The main training loop
    │   ├── evaluation # Evaluators for testing the model 
    │   ├── __init__.py
    │   ├── layers # Widely-used neural network modules
    │   ├── modeling # Definition of models
    │   │   ├── backbone # Vision backbone models
    │   │   ├── meta_arch
    │   │   │   ├── person_search # Person search models
    │   │   │   └── ......
    │   │   ├── proposal_generator # RPN and its variants
    │   │   ├── roi_heads # Prediction modules
    │   │   ├── transformer # Transformer models
    │   │   └── ......
    │   ├── model_zoo
    │   ├── solver # Optimizers and lr schedulers
    │   ├── structures # Commonly used data structures
    │   └── utils # Commonly used tools, e.g. logging, visualization and distributed communication.
    ├── README.md # This file
    ├── test.sh # Final test script
    ├── tools # Train/test interface and preprocess tools
    │   ├── convert_crowdhuman_to_coco.py
    │   ├── convert_swin_weight.py
    │   └── train_ps_net.py
    └── train.sh # Overall training script

    ```

## Acknowledgement

This code is greatly inspired by [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html).
