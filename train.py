import argparse
import os
import sys
import pandas as pd
sys.path.append(os.getcwd())
from pggan import PGGAN
from image_sampler import ImageSampler
from noise_sampler import NoiseSampler
from utils.config import args_to_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=str)
    parser.add_argument('--channel', '-c', type=int, default=3)
    parser.add_argument('--nb_growing', '-g', type=int, default=8)
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--nb_epoch', '-e', type=int, default=1000)
    parser.add_argument('--latent_dim', '-ld', type=int, default=100)
    parser.add_argument('--save_steps', '-ss', type=int, default=1)
    parser.add_argument('--visualize_steps', '-vs', type=int, default=1)
    parser.add_argument('--logdir', '-log', type=str, default="../logs")
    parser.add_argument('--distribution', '-dis', type=str, default="uniform")
    parser.add_argument('--upsampling', '-up', type=str, default="deconv")
    parser.add_argument('--downsampling', '-down', type=str, default="stride")
    parser.add_argument('--lr_d', type=float, default=1e-4)
    parser.add_argument('--lr_g', type=float, default=1e-4)
    parser.add_argument('--gp_lambda', '-lmd', type=float, default=10.)
    parser.add_argument('--d_norm_eps', '-eps', type=float, default=1e-3)

    args = parser.parse_args()

    args_to_csv(os.path.join(args.logdir, 'config.csv'), args)

    image_sampler = ImageSampler()
    noise_sampler = NoiseSampler(args.distribution)

    model = PGGAN(channel=args.channel,
                  latent_dim=args.latent_dim,
                  nb_growing=args.nb_growing,
                  gp_lambda=args.gp_lambda,
                  d_norm_eps=args.d_norm_eps,
                  upsampling=args.upsampling,
                  downsampling=args.downsampling,
                  lr_d=args.lr_d,
                  lr_g=args.lr_g)
    model.fit(image_sampler.flow_from_directory(args.image_dir,
                                                args.batch_size,
                                                with_class=False),
              noise_sampler,
              nb_epoch=args.nb_epoch,
              logdir=args.logdir,
              save_steps=args.save_steps,
              visualize_steps=args.visualize_steps)


if __name__ == '__main__':
    main()