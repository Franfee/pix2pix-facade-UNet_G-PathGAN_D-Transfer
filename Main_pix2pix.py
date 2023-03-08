import os
import sys
import time
import datetime
import numpy as np

from torchvision.utils import save_image

from net.models import *
from utils.get_dataLoader import opt, get_data_loader

# check output dir
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

# dataLoader
train_dataloader = get_data_loader("train")
val_dataloader = get_data_loader("test")

# Tensor type
use_cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# Initialize generator and discriminator
# 论文 3.2. Network architectures: G和D的modules块形式为: "卷积-归一化-激活"
# 论文 3.2.1 Generator with skips : 很多工作是bottleneck结构:输入通过一系列逐步向下采样的层，直到瓶颈层，在这一点上，过程被反转
# 对于许多图像转换问题，在输入和输出之间有大量的低级信息共享，并且在网络上直接传递这些信息是可取的。例如，在图像着色的情况下，输入和输出共享突出边缘的位置。
# 论文Fig3:生成器的可用的架构有两种,U-Net是具有(对应大小层的) skip connections. (相当于AE+ResNet)
Generator = GeneratorUNet()
# 因此，我们设计了一个鉴别器体系结构——我们称之为PatchGAN——它只在patch的规模上惩罚结构。
Discriminator = Discriminator()

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)  # 256 // (2^4) = 16


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    val_img_pair = next(iter(val_dataloader))
    real_A = val_img_pair["B"].type(Tensor)
    real_B = val_img_pair["A"].type(Tensor)
    fake_B = Generator(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, "images/%s/step_%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)


def Train():
    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(train_dataloader):
            # Model inputs.  batch["A"] is origin-img ,batch["B"] is label-img
            real_src = batch["B"].type(Tensor)
            real_tgt = batch["A"].type(Tensor)

            # Adversarial ground truths
            valid = Tensor(np.ones((real_src.size(0), *patch)))
            fake = Tensor(np.zeros((real_src.size(0), *patch)))

            # ----------------Train Generators------------------
            optimizer_G.zero_grad()

            # GAN loss
            fake_src = Generator(real_src)
            pred_fake = Discriminator(fake_src, real_src)
            loss_GAN = criterion_GAN_MSE(pred_fake, valid)
            # Pixel-wise loss
            loss_pixel = criterion_PixelWise_L1(fake_src, real_tgt)
            # Total loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel
            loss_G.backward()
            optimizer_G.step()
            # --------------------------------------------------

            # ------------------Train Discriminator-------------------
            optimizer_D.zero_grad()

            # Real loss
            pred_real = Discriminator(real_tgt, real_src)
            loss_real = criterion_GAN_MSE(pred_real, valid)
            # Fake loss
            pred_fake = Discriminator(fake_src.detach(), real_src)
            loss_fake = criterion_GAN_MSE(pred_fake, fake)
            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()
            # --------------------------------------------------

            # ------------Log Progress------------
            # Determine approximate time left
            batches_done = epoch * len(train_dataloader) + i
            batches_left = opt.n_epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                % (epoch + 1, opt.n_epochs, i + 1, len(train_dataloader), loss_D.item(),
                   loss_G.item(), loss_pixel.item(), loss_GAN.item(), time_left)
            )
            # ----------------------------------

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)
            # end one batch
        # end on epoch

        # Save model checkpoints
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            torch.save(Generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
            torch.save(Discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))
    # end all epoch
    torch.save(Generator.state_dict(), "saved_models/%s/generator_final.pth" % opt.dataset_name)
    torch.save(Discriminator.state_dict(), "saved_models/%s/discriminator__final.pth" % opt.dataset_name)


if __name__ == '__main__':
    if opt.epoch != 0:
        # Load pretrained models
        Generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
        Discriminator.load_state_dict(
            torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
    else:
        # Initialize weights
        Generator.apply(weights_init_normal)
        Discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(Generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(Discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Loss functions
    criterion_GAN_MSE = torch.nn.MSELoss()
    criterion_PixelWise_L1 = torch.nn.L1Loss()  # page4: 3.2.2 L1 will already do.

    # Loss weight of L1 pixel-wise loss between translated image and real image
    lambda_pixel = 100

    # use GPU
    if use_cuda:
        Generator = Generator.cuda()
        Discriminator = Discriminator.cuda()
        criterion_GAN_MSE.cuda()
        criterion_PixelWise_L1.cuda()

    # start train
    Train()
