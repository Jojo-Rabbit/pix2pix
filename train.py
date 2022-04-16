import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from discriminator import Discriminator
from generator import Generator
from dataloader import AnimeDataset, MapDataset
from utils import *

torch.backends.cudnn.benchmark = True


def train(disc, gen, loader, opt_disc, opt_gen, bce, l1, g_scaler, d_scaler):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            d_fake = disc(x, y_fake.detach())
            d_real = disc(x, y)
            d_fake_loss = bce(d_fake, torch.zeros_like(d_fake))
            d_real_loss = bce(d_real, torch.ones_like(d_real))
            d_loss = (d_real_loss + d_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with torch.cuda.amp.autocast():
            d_fake = disc(x, y_fake)
            g_fake_loss = bce(d_fake, torch.ones_like(d_fake))
            l1_loss = l1(y_fake, y) * config.L1_LAMBDA
            g_loss = g_fake_loss + l1_loss

        gen.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                d_real=torch.sigmoid(d_real).mean().item(),
                d_fake=torch.sigmoid(d_fake).mean().item(),
            )


def main():
    disc = Discriminator(in_channels=config.CHANNELS_IMG).to(config.DEVICE)
    gen = Generator(in_channel=config.CHANNELS_IMG).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1 = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    if config.DATASET == 'anime':
        train_dataset = AnimeDataset(config.ROOT_DIR)
    else:
        train_dataset = MapDataset(config.ROOT_DIR)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    if config.DATASET == 'anime':
        val_dataset = AnimeDataset(config.VAL_DIR)
    else:
        val_dataset = MapDataset(config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    for epoch in range(config.NUM_EPOCHS):
        train(disc, gen, train_loader, opt_disc, opt_gen, BCE, L1, g_scaler, d_scaler)

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)

        save_some_examples(gen, val_loader, epoch, folder='/content/data')


if __name__ == "__main__":
    main()