# distill_visual_priors
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/distilling-visual-priors-from-self-supervised/object-classification-on-imagenet-vipriors)](https://paperswithcode.com/sota/object-classification-on-imagenet-vipriors?p=distilling-visual-priors-from-self-supervised)

This is the 2nd place solution of ECCV 2020 workshop VIPriors Image Classification Challenge.

![U6i0Pg.png](https://s1.ax1x.com/2020/07/17/U6i0Pg.png)

The two phases of our proposed method. The first phase is to construct a useful visual prior with self-supervised contrastive learning, and the second phase is to perform self-distillation on the pre-trained checkpoint. The student model is trained with a distillation loss and a classification loss, while the teacher model is frozen.	


## Usage

Our solution presents a two-phase pipeline, and **we only use the provided subset of ImageNet, no external data or checkpoint is used in our solution.**

### Phase-1

Self-supervised pretraining.

Please follow the instructions in the `moco` folder.

### Phase-2

Self-distillation and classification finetuning.

```bash
cd sup_train_distill
python3 train_selfsup.py --data_path /path/to/data/ --net_type self_sup_r50 --input-res 448 --pretrained /path/to/unsupervise_pretrained_checkpoint --save_path /path/to/save --batch_size 256 --autoaug --label_smooth
```

## Citations

Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follow.

```
@inproceedings{zhao2020distilling,
  title={Distilling visual priors from self-supervised learning},
  author={Zhao, Bingchen and Wen, Xin},
  booktitle={European Conference on Computer Vision},
  pages={422--429},
  year={2020},
  organization={Springer}
}
```


## Contact

Bingchen Zhao: zhaobc.gm@gmail.com

