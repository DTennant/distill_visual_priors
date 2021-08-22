## MoCo: Momentum Contrast for Unsupervised Visual Representation Learning


### Preparation

1. Install PyTorch
2. Follow the instructions in [Competition toolkits](https://github.com/VIPriors/vipriors-challenges-toolkit/tree/master/image-classification) to set up the dataset.


### Self-supervised Pre-training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To conduct self-supervised pre-training with a ResNet-50 model on ImageNet on an 8-gpu machine, run:
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 --smallbank \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your dataset-folder with train and val folders]
```
This script uses all the default hyper-parameters as described in the MoCo v1 paper. To run MoCo v2, set `--mlp --moco-t 0.2 --aug-plus --cos`.


### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

