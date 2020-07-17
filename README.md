# distill_visual_priors

This is the 3rd place solution of ECCV 2020 workshop VIPriors Image Classification Challenge.

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

## Contact

Bingchen Zhao: zhaobc.gm@gmail.com

