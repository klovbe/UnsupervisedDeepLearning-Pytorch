import sys
sys.path.append("..")
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from udlp.clustering.vade import VaDE
import argparse
from data_prcoess import mydataset
from time import time

parser = argparse.ArgumentParser(description='VAE + GMM for single cell data clustering')
parser.add_argument('--lr', type=float, default=0.002, metavar='N',
                    help='learning rate for training (default: 0.002)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--pretrain', type=str, default="model/pretrained_vade-3layer.pt", metavar='N',
                    help='path of pretrained ae model')
parser.add_argument('--save', type=str, default="", metavar='N',
                    help='path to save model')
parser.add_argument('--datapath', type=str, default="", metavar='N',
                    help='path of data')
parser.add_argument('--labelpath', type=str, default="", metavar='N',
                    help='path of label')
parser.add_argument('--input_dim', type=int, default="", metavar='N',
                    help='input dimension of encoder')
parser.add_argument('--n_centroids', type=int, default="", metavar='N',
                    help='number of clusters')
args = parser.parse_args()

train_loader = torch.utils.data.DataLoader(
    mydataset(args.datapath, args.labelpath, args.input_dim, transform=transforms.ToTensor(),
              target_transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data/mnist', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, num_workers=2)

#pretrain


t0 = time()
vade = VaDE(input_dim=args.input_dim, z_dim=10, n_centroids=args.n_centroids, binary=False,
        encodeLayer=[500,500,2000], decodeLayer=[2000,500,500])
if args.pretrain != "":
    print("Loading model from %s..." % args.pretrain)
    vade.load_model(args.pretrain)
print("Initializing through GMM..")
vade.initialize_gmm(train_loader)
vade.fit(train_loader, lr=args.lr, batch_size=args.batch_size, num_epochs=args.epochs, anneal=True)
print("clustering time: ", (time() - t0))
if args.save != "":
    vade.save_model(args.save)