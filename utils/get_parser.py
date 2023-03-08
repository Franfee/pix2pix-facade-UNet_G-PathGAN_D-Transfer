# -*- coding: utf-8 -*-
# @Time    : 2023/3/7 9:54
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12

import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="facade", help="name of the dataset")

    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval of sampling images from generators")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval of model checkpoints")

    opt = parser.parse_args()
    print(opt)
    return opt
