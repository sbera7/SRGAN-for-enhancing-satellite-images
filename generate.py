import torch
import config
from torch import nn
from torch import optim
from utils import load_checkpoint, save_checkpoint, plot_examples
from loss import VGGLoss
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from tqdm import tqdm
from dataset import MyImageFolder

torch.backends.cudnn.benchmark = True

model = Generator(in_channels=3).to(config.DEVICE)
checkpoint = torch.load(config.CHECKPOINT_GEN, map_location=config.DEVICE)
model.load_state_dict(checkpoint["state_dict"])
plot_examples("lr_test/", model)