#!/bin/bash
#CUDA_VISIBLE_DEVICES=0 python gen_optflow2018.py --train_dir /media/SSD1/EndoVis2018_annotated/data_endovis2018/train
CUDA_VISIBLE_DEVICES=3 python gen_optflow2018.py --train_dir /media/SSD1/EndoVis2018_annotated/data_endovis2018/val
