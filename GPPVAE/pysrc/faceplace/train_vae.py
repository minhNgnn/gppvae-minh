import matplotlib
import sys

matplotlib.use("Agg")
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vae import FaceVAE
import h5py
import os
import pdb
import logging
import pylab as pl
from utils import smartSum, smartAppendDict, smartAppend, export_scripts
from callbacks import callback
from data_parser import read_face_data, FaceDataset
from optparse import OptionParser
import logging
import pickle
import time
import wandb
import yaml
from datetime import datetime


def load_config(yaml_path=None):
    """Load configuration from YAML file."""
    if yaml_path is None:
        # Use config.yml in the same directory as this file
        config_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(config_dir, "config.yml")
    
    # Default config
    config = {
        'data': "./../data/faceplace/data_faces.h5",
        'outdir': "./../out/vae",
        'seed': 0,
        'filts': 32,
        'zdim': 256,
        'vy': 0.002,
        'lr': 0.0002,
        'bs': 64,
        'epoch_cb': 100,
        'epochs': 10000,
        'debug': False,
        'use_wandb': False,
        'wandb_project': 'gppvae',
        'wandb_run_name': None,
    }
    
    # Load from YAML if exists
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Flatten YAML structure for VAE
        if 'data' in yaml_config and 'path' in yaml_config['data']:
            config['data'] = yaml_config['data']['path']
        if 'output' in yaml_config and 'vae_dir' in yaml_config['output']:
            config['outdir'] = yaml_config['output']['vae_dir']
        if 'vae' in yaml_config:
            if 'filts' in yaml_config['vae']:
                config['filts'] = yaml_config['vae']['filts']
            if 'zdim' in yaml_config['vae']:
                config['zdim'] = yaml_config['vae']['zdim']
        if 'loss' in yaml_config and 'vy' in yaml_config['loss']:
            config['vy'] = yaml_config['loss']['vy']
        if 'training' in yaml_config:
            if 'vae_lr' in yaml_config['training']:
                config['lr'] = yaml_config['training']['vae_lr']
            if 'batch_size' in yaml_config['training']:
                config['bs'] = yaml_config['training']['batch_size']
            if 'epochs' in yaml_config['training']:
                config['epochs'] = yaml_config['training']['epochs']
            if 'seed' in yaml_config['training']:
                config['seed'] = yaml_config['training']['seed']
        if 'logging' in yaml_config and 'epoch_callback' in yaml_config['logging']:
            config['epoch_cb'] = yaml_config['logging']['epoch_callback']
        if 'wandb' in yaml_config:
            if 'enabled' in yaml_config['wandb']:
                config['use_wandb'] = yaml_config['wandb']['enabled']
            if 'project' in yaml_config['wandb']:
                config['wandb_project'] = yaml_config['wandb']['project']
            if 'run_name' in yaml_config['wandb']:
                config['wandb_run_name'] = yaml_config['wandb']['run_name']
        if 'debug' in yaml_config:
            config['debug'] = yaml_config['debug']
    
    return config


parser = OptionParser()
parser.add_option(
    "--data",
    dest="data",
    type=str,
    default=None,
    help="dataset path (overrides config.yml)",
)
parser.add_option(
    "--outdir", dest="outdir", type=str, default=None, help="output dir (overrides config.yml)"
)
parser.add_option("--seed", dest="seed", type=int, default=None, help="random seed")
parser.add_option(
    "--filts", dest="filts", type=int, default=None, help="number of convolutional filters"
)
parser.add_option("--zdim", dest="zdim", type=int, default=None, help="latent dimension size")
parser.add_option(
    "--vy", dest="vy", type=float, default=None, help="observation noise variance"
)
parser.add_option("--lr", dest="lr", type=float, default=None, help="learning rate")
parser.add_option("--bs", dest="bs", type=int, default=None, help="batch size")
parser.add_option(
    "--epoch_cb",
    dest="epoch_cb",
    type=int,
    default=None,
    help="callback frequency in epochs",
)
parser.add_option(
    "--epochs", dest="epochs", type=int, default=None, help="total number of epochs"
)
parser.add_option("--debug", action="store_true", dest="debug", default=False)
parser.add_option(
    "--use_wandb", action="store_true", dest="use_wandb", default=False, help="use weights and biases logging"
)
parser.add_option(
    "--wandb_project", dest="wandb_project", type=str, default=None, help="wandb project name"
)
parser.add_option(
    "--wandb_run_name", dest="wandb_run_name", type=str, default=None, help="wandb run name"
)
(args, _) = parser.parse_args()

# Load config from YAML
opt = load_config()

# Override with command-line arguments
if args.data is not None:
    opt['data'] = args.data
if args.outdir is not None:
    opt['outdir'] = args.outdir
if args.seed is not None:
    opt['seed'] = args.seed
if args.filts is not None:
    opt['filts'] = args.filts
if args.zdim is not None:
    opt['zdim'] = args.zdim
if args.vy is not None:
    opt['vy'] = args.vy
if args.lr is not None:
    opt['lr'] = args.lr
if args.bs is not None:
    opt['bs'] = args.bs
if args.epochs is not None:
    opt['epochs'] = args.epochs
if args.epoch_cb is not None:
    opt['epoch_cb'] = args.epoch_cb
if args.debug:
    opt['debug'] = args.debug
if args.use_wandb:
    opt['use_wandb'] = args.use_wandb
if args.wandb_project is not None:
    opt['wandb_project'] = args.wandb_project
if args.wandb_run_name is not None:
    opt['wandb_run_name'] = args.wandb_run_name


# Add timestamp to output directory for unique runs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_outdir = opt['outdir']
opt['outdir'] = os.path.join(base_outdir, timestamp)

if not os.path.exists(opt['outdir']):
    os.makedirs(opt['outdir'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# output dir
wdir = os.path.join(opt['outdir'], "weights")
fdir = os.path.join(opt['outdir'], "plots")
if not os.path.exists(wdir):
    os.makedirs(wdir)
if not os.path.exists(fdir):
    os.makedirs(fdir)

# copy code to output folder
export_scripts(os.path.join(opt['outdir'], "scripts"))

# create logfile
log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(opt['outdir'], "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("opt = %s", opt)

# extract VAE settings and export
vae_cfg = {"nf": opt['filts'], "zdim": opt['zdim'], "vy": opt['vy']}
pickle.dump(vae_cfg, open(os.path.join(opt['outdir'], "vae.cfg.p"), "wb"))


def main():

    torch.manual_seed(opt['seed'])

    if opt['debug']:
        pdb.set_trace()

    # initialize wandb
    if opt['use_wandb']:
        wandb.init(
            project=opt['wandb_project'],
            name=opt['wandb_run_name'],
            config=opt
        )

    # define VAE and optimizer
    vae = FaceVAE(**vae_cfg).to(device)

    # optimizer
    optimizer = optim.Adam(vae.parameters(), lr=opt['lr'])

    # load data
    img, obj, view = read_face_data(opt['data'])  # image, object, and view
    train_data = FaceDataset(img["train"], obj["train"], view["train"])
    val_data = FaceDataset(img["val"], obj["val"], view["val"])
    train_queue = DataLoader(train_data, batch_size=opt['bs'], shuffle=True)
    val_queue = DataLoader(val_data, batch_size=opt['bs'], shuffle=False)

    history = {}
    for epoch in range(opt['epochs']):

        # train and eval
        ht = train_ep(vae, train_queue, optimizer)
        hv = eval_ep(vae, val_queue)
        smartAppendDict(history, ht)
        smartAppendDict(history, hv)
        logging.info(
            "epoch %d - train_mse: %f - test_mse %f" % (epoch, ht["mse"], hv["mse_val"])
        )

        # log to wandb
        if opt['use_wandb']:
            wandb.log({
                "epoch": epoch,
                "train/mse": ht["mse"],
                "train/nll": ht["nll"],
                "train/kld": ht["kld"],
                "train/loss": ht["loss"],
                "val/mse": hv["mse_val"],
                "val/nll": hv["nll_val"],
                "val/kld": hv["kld_val"],
                "val/loss": hv["loss_val"],
            })

        # callbacks
        if epoch % opt['epoch_cb'] == 0:
            logging.info("epoch %d - executing callback" % epoch)
            wfile = os.path.join(wdir, "weights.%.5d.pt" % epoch)
            ffile = os.path.join(fdir, "plot.%.5d.png" % epoch)
            torch.save(vae.state_dict(), wfile)
            callback(epoch, val_queue, vae, history, ffile, device)
            
            # log plot to wandb
            if opt['use_wandb']:
                wandb.log({"reconstructions": wandb.Image(ffile)})

    # finish wandb run
    if opt['use_wandb']:
        wandb.finish()


def train_ep(vae, train_queue, optimizer):

    rv = {}
    vae.train()

    for batch_i, data in enumerate(train_queue):

        # forward
        y = data[0]
        eps = Variable(torch.randn(y.shape[0], 256), requires_grad=False)
        y, eps = y.to(device), eps.to(device)
        elbo, mse, nll, kld = vae.forward(y, eps)
        loss = elbo.sum()

        # back propagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # sum metrics
        _n = train_queue.dataset.Y.shape[0]
        smartSum(rv, "mse", float(mse.data.sum().cpu()) / float(_n))
        smartSum(rv, "nll", float(nll.data.sum().cpu()) / float(_n))
        smartSum(rv, "kld", float(kld.data.sum().cpu()) / float(_n))
        smartSum(rv, "loss", float(elbo.data.sum().cpu()) / float(_n))

    return rv


def eval_ep(vae, val_queue):
    rv = {}
    vae.eval()

    with torch.no_grad():

        for batch_i, data in enumerate(val_queue):

            # forward
            y = data[0]
            eps = Variable(torch.randn(y.shape[0], 256), requires_grad=False)
            y, eps = y.to(device), eps.to(device)
            elbo, mse, nll, kld = vae.forward(y, eps)

            # sum metrics
            _n = val_queue.dataset.Y.shape[0]
            smartSum(rv, "mse_val", float(mse.data.sum().cpu()) / float(_n))
            smartSum(rv, "nll_val", float(nll.data.sum().cpu()) / float(_n))
            smartSum(rv, "kld_val", float(kld.data.sum().cpu()) / float(_n))
            smartSum(rv, "loss_val", float(elbo.data.sum().cpu()) / float(_n))

    return rv


if __name__ == "__main__":
    main()
