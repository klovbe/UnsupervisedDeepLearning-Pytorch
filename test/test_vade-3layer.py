import sys
sys.path.append("..")
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from udlp.clustering.vade import VaDE
from udlp.clustering.vade_bn import VaDE_bn
import argparse
from data_prcoess import mydataset
from time import time
from udlp.autoencoder.stackedDAE import StackedDAE
import os


parser = argparse.ArgumentParser(description='VAE + GMM for single cell data clustering')
parser.add_argument('--lr', type=float, default=0.002, metavar='N',
                    help='learning rate for training (default: 0.002)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                    help='number of epochs to train (default: 500)')
# parser.add_argument('--pretrain', type=str, default="model/pretrained_vade-3layer.pt", metavar='N',
#                     help='path of pretrained ae model')
# parser.add_argument('--save', type=str, default="", metavar='N',
#                     help='path to save model')
parser.add_argument('--model_name', type=str, default="", metavar='N',
                    help='name of data')
# parser.add_argument('--input_dim', type=int, default="", metavar='N',
#                     help='input dimension of encoder')
# parser.add_argument('--n_centroids', type=int, default="", metavar='N',
#                     help='number of clusters')
parser.add_argument('--save_inter', type=int, default=100, metavar='N',
                    help='interval to save metrics')
parser.add_argument('--gene_select', type=int, default=1000, metavar='N',
                    help='gene selected for training')
parser.add_argument('--pretrain_epochs', type=int, default=200, metavar='N',
                    help='pretrian epochs')
args = parser.parse_args()

Mydataset = mydataset(args.model_name, args.gene_select,gene_select=args.gene_select, transform=None,
              target_transform=None)
# cell,target = Mydataset.__getitem__(1)
train_loader = torch.utils.data.DataLoader(Mydataset,
    batch_size=args.batch_size, shuffle=True)
n_centroids = Mydataset.get_n_centroids()
print("the data has {} clusters".format(n_centroids))

#pretrain
pretrain_path = 'model/'+args.model_name+'_pretrain.pt'
if os.path.exists(pretrain_path) is False:
    sdae = StackedDAE(input_dim=args.gene_select, z_dim=10, binary=False,
                  encodeLayer=[300, 100,  30], decodeLayer=[30,  100, 300], activation="relu",
                  dropout=0, is_bn=False)
    sdae.fit(train_loader, lr=args.lr, num_epochs=args.pretrain_epochs, corrupt=0.3)
    sdae.save_model(pretrain_path)
else:
    print('pretrained model exists')


t0 = time()
vade = VaDE(input_dim=args.gene_select, z_dim=10, n_centroids=n_centroids, binary=False,
        encodeLayer=[300,100,30], decodeLayer=[30,100,300], activation="relu",
             dropout=0, is_bn=False)
if os._exists(pretrain_path):
    print("Loading model from %s..." % pretrain_path)
    vade.load_model(pretrain_path)
print("Initializing through GMM..")
vade.initialize_gmm(train_loader)
print("basline of GMM and kmeans")
vade.gmm_kmeans_cluster(train_loader)
vade.fit(train_loader, model_name=args.model_name, save_inter=args.save_inter, lr=args.lr, batch_size=args.batch_size,
         num_epochs=args.epochs, anneal=True)
print("clustering time: ", (time() - t0))
save_path = 'model/'+args.model_name+'.pt'
vade.save_model(save_path)