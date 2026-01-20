import sys

sys.path.append("./..")
import pdb
import os
import numpy as np
import pylab as pl
import torch
from torch.autograd import Variable
from matplotlib.gridspec import GridSpec


def _compose(orig, recon):
    _imgo = []
    _imgr = []
    for i in range(orig.shape[0]):
        _imgo.append(orig[i])
    for i in range(orig.shape[0]):
        _imgr.append(recon[i])
    _imgo = np.concatenate(_imgo, 1)
    _imgr = np.concatenate(_imgr, 1)
    _rv = np.concatenate([_imgo, _imgr], 0)
    _rv = np.clip(_rv, 0, 1)
    return _rv


def _compose_multi(imgs):
    _imgs = []
    for i in range(len(imgs)):
        _imgs.append([])
        for j in range(imgs[i].shape[0]):
            _imgs[i].append(imgs[i][j])
        _imgs[i] = np.concatenate(_imgs[i], 1)
    _rv = np.concatenate(_imgs, 0)
    _rv = np.clip(_rv, 0, 1)
    return _rv


def callback(epoch, val_queue, vae, history, figname, device):

    with torch.no_grad():

        # compute z
        zm = []
        zs = []
        for batch_i, data in enumerate(val_queue):
            y = data[0].to(device)
            _zm, _zs = vae.encode(y)
            zm.append(_zm.data.cpu().numpy())
            zs.append(_zs.data.cpu().numpy())
        zm, zs = np.concatenate(zm, 0), np.concatenate(zs, 0)

        # init fig with proper size and clear any existing figure
        pl.close('all')
        fig = pl.figure(figsize=(12, 10))
        
        # Use GridSpec for better layout control
        # Left side: 4 rows x 2 cols for metrics (columns 0-1)
        # Right side: 4 rows x 1 col for images (column 2)
        gs = GridSpec(4, 3, figure=fig, width_ratios=[1, 1, 1.5], 
                      hspace=0.4, wspace=0.3,
                      left=0.08, right=0.95, top=0.95, bottom=0.05)

        # plot history - left side (2x2 grid in first 2 columns)
        xs = np.arange(1, epoch + 2)
        keys = ["loss", "nll", "kld", "mse"]
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for ik, key in enumerate(keys):
            row, col = positions[ik]
            ax = fig.add_subplot(gs[row, col])
            ax.set_title(key, fontsize=10)
            ax.plot(xs, history[key], "k", label="train")
            if key not in ["lr", "vy"]:
                ax.plot(xs, history[key + "_val"], "r", label="val")
            if key == "mse":
                ax.set_ylim(0.0, 0.01)
            ax.tick_params(labelsize=8)

        # plot hist of zm and zs - bottom left
        ax_zm = fig.add_subplot(gs[2, 0])
        ax_zm.set_title("Zm", fontsize=10)
        _y, _x = np.histogram(zm.ravel(), 30)
        _x = 0.5 * (_x[:-1] + _x[1:])
        ax_zm.plot(_x, _y, "k")
        ax_zm.tick_params(labelsize=8)
        
        ax_zs = fig.add_subplot(gs[2, 1])
        ax_zs.set_title("log$_{10}$Zs", fontsize=10)
        _y, _x = np.histogram(np.log10(zs.ravel()), 30)
        _x = 0.5 * (_x[:-1] + _x[1:])
        ax_zs.plot(_x, _y, "k")
        ax_zs.tick_params(labelsize=8)

        # Sample diverse validation samples for visualization
        n_val = val_queue.dataset.Y.shape[0]
        if n_val >= 24:
            sample_stride = max(1, n_val // 24)
            sample_indices = np.arange(0, n_val, sample_stride)[:24]
        else:
            sample_indices = np.arange(min(24, n_val))
        
        # val reconstructions with diverse sampling
        _zm = Variable(torch.tensor(zm[sample_indices]), requires_grad=False).to(device)
        Rv = vae.decode(_zm).data.cpu().numpy().transpose((0, 2, 3, 1))
        Yv = val_queue.dataset.Y[sample_indices].numpy().transpose((0, 2, 3, 1))

        # make image plots on right column - 4 rows
        # Each shows 6 original images (top) and 6 reconstructions (bottom)
        for row_idx, (start, end) in enumerate([(0, 6), (6, 12), (12, 18), (18, 24)]):
            ax = fig.add_subplot(gs[row_idx, 2])
            if end <= len(Yv):
                _img = _compose(Yv[start:end], Rv[start:end])
                ax.imshow(_img)
            ax.axis('off')  # Remove axis ticks and labels for image plots
            if row_idx == 0:
                ax.set_title("Original (top) vs Recon (bottom)", fontsize=9)

        pl.savefig(figname, dpi=150, bbox_inches='tight')
        pl.close('all')


def callback_gppvae0(epoch, history, covs, imgs, ffile):

    # Close any existing figures to prevent overlap
    pl.close('all')
    
    # init fig with proper size
    fig = pl.figure(figsize=(12, 10))
    
    # Use GridSpec for better layout control
    gs = GridSpec(4, 3, figure=fig, width_ratios=[1, 1, 1.5], 
                  hspace=0.35, wspace=0.35,
                  left=0.08, right=0.95, top=0.94, bottom=0.05)
    
    # Row 0, Col 0: loss
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.set_title("loss", fontsize=10)
    ax_loss.plot(history["loss"], "k")
    ax_loss.tick_params(labelsize=8)
    
    # Row 0, Col 1: vars
    ax_vars = fig.add_subplot(gs[0, 1])
    ax_vars.set_title("vars", fontsize=10)
    ax_vars.plot(np.array(history["vs"])[:, 0], "r", label="v0")
    ax_vars.plot(np.array(history["vs"])[:, 1], "k", label="v1")
    ax_vars.tick_params(labelsize=8)
    
    # Row 1, Col 0: mse_out
    ax_mse = fig.add_subplot(gs[1, 0])
    ax_mse.set_title("mse_out", fontsize=10)
    ax_mse.plot(history["mse_out"], "k")
    ax_mse.tick_params(labelsize=8)

    # Row 2, Col 0: XX covariance
    ax_xx = fig.add_subplot(gs[2, 0])
    ax_xx.set_title("XX (identity cov)", fontsize=10)
    im_xx = ax_xx.imshow(covs["XX"], vmin=-0.4, vmax=1, aspect='auto')
    pl.colorbar(im_xx, ax=ax_xx, fraction=0.046, pad=0.04)
    ax_xx.tick_params(labelsize=8)
    
    # Row 2, Col 1: WW covariance
    ax_ww = fig.add_subplot(gs[2, 1])
    ax_ww.set_title("WW (view cov)", fontsize=10)
    im_ww = ax_ww.imshow(covs["WW"], vmin=-0.4, vmax=1, aspect='auto')
    pl.colorbar(im_ww, ax=ax_ww, fraction=0.046, pad=0.04)
    ax_ww.tick_params(labelsize=8)

    Yv, Rv = imgs["Yv"], imgs["Yo"]

    # Right column: Image reconstructions (4 rows)
    for row_idx, (start, end) in enumerate([(0, 6), (6, 12), (12, 18), (18, 24)]):
        ax = fig.add_subplot(gs[row_idx, 2])
        if end <= len(Yv):
            _img = _compose(Yv[start:end], Rv[start:end])
            ax.imshow(_img)
        ax.axis('off')  # Remove all axis ticks and labels
        if row_idx == 0:
            ax.set_title("Original (top) vs Recon (bottom)", fontsize=9)

    pl.savefig(ffile, dpi=150, bbox_inches='tight')
    pl.close('all')


def callback_gppvae(epoch, history, covs, imgs, ffile):

    # Close any existing figures to prevent overlap
    pl.close('all')
    
    # init fig with proper size
    fig = pl.figure(figsize=(14, 10))
    
    # Use GridSpec for better layout control
    # Left side: metrics and covariance matrices (columns 0-1)
    # Right side: images (column 2)
    gs = GridSpec(4, 3, figure=fig, width_ratios=[1, 1, 1.8], 
                  hspace=0.35, wspace=0.35,
                  left=0.06, right=0.96, top=0.94, bottom=0.05)
    
    # Row 0: loss and vars
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.set_title("loss", fontsize=10)
    ax_loss.plot(history["loss"], "k")
    ax_loss.tick_params(labelsize=8)
    
    ax_vars = fig.add_subplot(gs[0, 1])
    ax_vars.set_title("vars", fontsize=10)
    ax_vars.plot(np.array(history["vs"])[:, 0], "r", label="v0")
    ax_vars.plot(np.array(history["vs"])[:, 1], "k", label="v1")
    ax_vars.set_ylim(0, 1)
    ax_vars.tick_params(labelsize=8)
    
    # Row 1: recon_term and gp_nll
    ax_recon = fig.add_subplot(gs[1, 0])
    ax_recon.set_title("recon_term", fontsize=10)
    ax_recon.plot(history["recon_term"], "k")
    ax_recon.tick_params(labelsize=8)
    
    ax_gp = fig.add_subplot(gs[1, 1])
    ax_gp.set_title("gp_nll", fontsize=10)
    ax_gp.plot(history["gp_nll"], "k")
    ax_gp.tick_params(labelsize=8)
    
    # Row 2: mse_out and mse_0
    ax_mse_out = fig.add_subplot(gs[2, 0])
    ax_mse_out.set_title("mse_out", fontsize=10)
    ax_mse_out.plot(history["mse_out"], "k")
    ax_mse_out.set_ylim(0, 0.1)
    ax_mse_out.tick_params(labelsize=8)
    
    ax_mse0 = fig.add_subplot(gs[2, 1])
    ax_mse0.set_title("mse_0", fontsize=10)
    ax_mse0.plot(history["mse"], "k", label="train")
    ax_mse0.plot(history["mse_val"], "r", label="val")
    ax_mse0.set_ylim(0, 0.01)
    ax_mse0.tick_params(labelsize=8)
    
    # Row 3: Covariance matrices
    ax_xx = fig.add_subplot(gs[3, 0])
    ax_xx.set_title("XX (identity cov)", fontsize=10)
    im_xx = ax_xx.imshow(covs["XX"], vmin=-0.4, vmax=1, aspect='auto')
    pl.colorbar(im_xx, ax=ax_xx, fraction=0.046, pad=0.04)
    ax_xx.tick_params(labelsize=8)
    
    ax_ww = fig.add_subplot(gs[3, 1])
    ax_ww.set_title("WW (view cov)", fontsize=10)
    im_ww = ax_ww.imshow(covs["WW"], vmin=-0.4, vmax=1, aspect='auto')
    pl.colorbar(im_ww, ax=ax_ww, fraction=0.046, pad=0.04)
    ax_ww.tick_params(labelsize=8)

    # Get images
    Yv, Yr, Rv = imgs["Yv"], imgs["Yr"], imgs["Yo"]

    # Sample diverse identities across the validation set
    n_total = Yv.shape[0]
    if n_total >= 24:
        sample_stride = max(1, n_total // 24)
        sample_indices = np.arange(0, n_total, sample_stride)[:24]
        Yv = Yv[sample_indices]
        Yr = Yr[sample_indices]
        Rv = Rv[sample_indices]

    # Right column: Image reconstructions (4 rows)
    for row_idx, (start, end) in enumerate([(0, 6), (6, 12), (12, 18), (18, 24)]):
        ax = fig.add_subplot(gs[row_idx, 2])
        if end <= len(Yv):
            _img = _compose_multi([Yv[start:end], Yr[start:end], Rv[start:end]])
            ax.imshow(_img)
        ax.axis('off')  # Remove all axis ticks and labels
        if row_idx == 0:
            ax.set_title("Ground Truth | VAE Recon | GP Pred", fontsize=9)

    pl.savefig(ffile, dpi=150, bbox_inches='tight')
    pl.close('all')
