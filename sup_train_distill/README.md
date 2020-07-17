## Phase-2

Self-distillation and classification finetuning on the pre-trained checkpoint.

Usage:

```bash
cd sup_train_distill
python3 train_selfsup.py --data_path /path/to/data/ --net_type self_sup_r50 --input-res 448 --pretrained /path/to/unsupervise_pretrained_checkpoint --save_path /path/to/save --batch_size 256 --autoaug --label_smooth
```
