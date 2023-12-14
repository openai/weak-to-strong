# A Simple Weak-to-Strong Experiment on ImageNet

We provide code for a simple weak-to-strong experiment on ImageNet. 
We generate the weak labels using an [AlexNet](https://pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html) model pretrained on ImageNet and we use linear probes on top of [DINO](https://github.com/facebookresearch/dino) models
as a strong student. 

The full training command:

```bash
python3 run_weak_strong.py \
    data_path: <DATA_PATH> \
    weak_model_name: <WEAK_MODEL>\
    strong_model_name: <STRONG_MODEL> \
    batch_size <BATCH_SIZE> \
    seed <SEED> \
    n_epochs <N_EPOCHS> \
    lr <LR> \
    n_train <N_TRAIN>
```
Parameters:

* ```DATA_PATH``` &mdash; path to the base directory containing ImageNet data, see [torchvision page](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageNet.html) for instructions; should contain files `ILSVRC2012_devkit_t12.tar.gz` and `ILSVRC2012_img_val.tar`
* ```WEAK_MODEL``` &mdash; weak model name:
    - `"alexnet"` is the only default model and the only one currently implemented
* ```STRONG_MODEL``` &mdash; weak model name:
    - `"resnet50_dino"` (default)
    - `"vitb8_dino"`
* ```BATCH_SIZE``` &mdash; batch size for weak label generation and embedding extraction (default: `128`)
* ```SEED``` &mdash; random seed for dataset shuffling (default: `0`)
* ```EPOCHS``` &mdash; number of training epochs (default: `10`)
* ```LR``` &mdash; initial learning rate (default: `1e-3`)
* ```N_TRAIN``` &mdash; number of datapoints used to train the linear probe; `50000 - N_TRAIN` datapoints are used as test (default: `40000`)



Example commands:

```bash
# AlexNet → ResNet50 (DINO):
python3 run_weak_strong.py --strong_model_name resnet50_dino --n_epochs 20

# AlexNet → ViT-B/8 (DINO):
python3 run_weak_strong.py --strong_model_name vitb8_dino --n_epochs 5
```

With the comands above we get the following results (note that the results may not reproduce exactly due to randomness):

| Model                   | Top-1 Accuracy |
|-------------------------|----------------|
| AlexNet                 | 56.6           |
| Dino ResNet50           | 63.7           |
| Dino ViT-B/8            | 74.9           |
| AlexNet → DINO ResNet50 | 60.7           |
| AlexNet → DINO ViT-B/8  | 64.2           |

You can add new custom models to the `models.py` and new datasets to `data.py`.