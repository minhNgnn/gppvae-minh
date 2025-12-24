import matplotlib
import sys

matplotlib.use("Agg")
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vae import FaceVAE
from vmod import Vmodel
from gp import GP
import h5py
import scipy as sp
import os
import pdb
import logging
import pylab as pl
from utils import smartSum, smartAppendDict, smartAppend, export_scripts
from callbacks import callback_gppvae
from data_parser import read_face_data, FaceDataset
from optparse import OptionParser
import logging
import pickle
import time
import wandb
import yaml


def load_config(yaml_path=None):
    """Load configuration from YAML file for GPPVAE."""
    if yaml_path is None:
        # Use config.yml in the same directory as this file
        config_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(config_dir, "config.yml")
    
    # Default config for GPPVAE
    config = {
        'data': "./../data/faceplace/data_faces.h5",
        'outdir': "./../out/gppvae",
        'vae_cfg': "./../out/vae/vae.cfg.p",
        'vae_weights': "./../out/vae/weights/weights.00000.pt",
        'seed': 0,
        'vae_lr': 0.0002,
        'gp_lr': 0.001,
        'xdim': 64,
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
        
        # Flatten YAML structure for GPPVAE
        if 'data' in yaml_config and 'path' in yaml_config['data']:
            config['data'] = yaml_config['data']['path']
        if 'output' in yaml_config and 'gppvae_dir' in yaml_config['output']:
            config['outdir'] = yaml_config['output']['gppvae_dir']
        if 'gppvae' in yaml_config:
            if 'xdim' in yaml_config['gppvae']:
                config['xdim'] = yaml_config['gppvae']['xdim']
            if 'vae_config' in yaml_config['gppvae']:
                config['vae_cfg'] = yaml_config['gppvae']['vae_config']
            if 'vae_weights' in yaml_config['gppvae']:
                config['vae_weights'] = yaml_config['gppvae']['vae_weights']
        if 'training' in yaml_config:
            if 'vae_lr' in yaml_config['training']:
                config['vae_lr'] = yaml_config['training']['vae_lr']
            if 'gp_lr' in yaml_config['training']:
                config['gp_lr'] = yaml_config['training']['gp_lr']
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
parser.add_option("--vae_cfg", dest="vae_cfg", type=str, default=None, help="path to VAE config")
parser.add_option("--vae_weights", dest="vae_weights", type=str, default=None, help="path to VAE weights")
parser.add_option("--seed", dest="seed", type=int, default=None, help="random seed")
parser.add_option(
    "--vae_lr",
    dest="vae_lr",
    type=float,
    default=None,
    help="learning rate of vae params",
)
parser.add_option(
    "--gp_lr", dest="gp_lr", type=float, default=None, help="learning rate of gp params"
)
parser.add_option(
    "--xdim", dest="xdim", type=int, default=None, help="rank of object linear covariance"
)
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
if args.vae_cfg is not None:
    opt['vae_cfg'] = args.vae_cfg
if args.vae_weights is not None:
    opt['vae_weights'] = args.vae_weights
if args.seed is not None:
    opt['seed'] = args.seed
if args.vae_lr is not None:
    opt['vae_lr'] = args.vae_lr
if args.gp_lr is not None:
    opt['gp_lr'] = args.gp_lr
if args.xdim is not None:
    opt['xdim'] = args.xdim
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

# Load VAE config
vae_cfg = pickle.load(open(opt['vae_cfg'], "rb"))

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

    # load data
    img, obj, view = read_face_data(opt['data'])  # image, object, and view
    train_data = FaceDataset(img["train"], obj["train"], view["train"])
    val_data = FaceDataset(img["val"], obj["val"], view["val"])
    train_queue = DataLoader(train_data, batch_size=opt['bs'], shuffle=True)
    val_queue = DataLoader(val_data, batch_size=opt['bs'], shuffle=False)

    # longint view and object repr
    Dt = Variable(obj["train"][:, 0].long(), requires_grad=False).cuda()
    Wt = Variable(view["train"][:, 0].long(), requires_grad=False).cuda()
    Dv = Variable(obj["val"][:, 0].long(), requires_grad=False).cuda()
    Wv = Variable(view["val"][:, 0].long(), requires_grad=False).cuda()

    # define VAE and optimizer
    vae = FaceVAE(**vae_cfg).to(device)
    RV = torch.load(opt['vae_weights'])
    vae.load_state_dict(RV)
    vae.to(device)

    # define gp
    P = sp.unique(obj["train"]).shape[0]
    Q = sp.unique(view["train"]).shape[0]
    vm = Vmodel(P, Q, opt['xdim'], Q).cuda()
    gp = GP(n_rand_effs=1).to(device)
    gp_params = nn.ParameterList()
    gp_params.extend(vm.parameters())
    gp_params.extend(gp.parameters())

    # define optimizers
    vae_optimizer = optim.Adam(vae.parameters(), lr=opt['vae_lr'])
    gp_optimizer = optim.Adam(gp_params, lr=opt['gp_lr'])

    if opt['debug']:
        pdb.set_trace()

    history = {}
    for epoch in range(opt['epochs']):

        # 1. encode Y in mini-batches
        Zm, Zs = encode_Y(vae, train_queue)

        # 2. sample Z
        Eps = Variable(torch.randn(*Zs.shape), requires_grad=False).cuda()
        Z = Zm + Eps * Zs

        # 3. evaluation step (not needed for training)
        Vt = vm(Dt, Wt).detach()
        Vv = vm(Dv, Wv).detach()
        rv_eval, imgs, covs = eval_step(vae, gp, vm, val_queue, Zm, Vt, Vv)

        # 4. compute first-order Taylor expansion coefficient
        Zb, Vbs, vbs, gp_nll = gp.taylor_coeff(Z, [Vt])
        rv_eval["gp_nll"] = float(gp_nll.data.mean().cpu()) / vae.K

        # 5. accumulate gradients over mini-batches and update params
        rv_back = backprop_and_update(
            vae,
            gp,
            vm,
            train_queue,
            Dt,
            Wt,
            Eps,
            Zb,
            Vbs,
            vbs,
            vae_optimizer,
            gp_optimizer,
        )
        rv_back["loss"] = (
            rv_back["recon_term"] + rv_eval["gp_nll"] + rv_back["pen_term"]
        )

        smartAppendDict(history, rv_eval)
        smartAppendDict(history, rv_back)
        smartAppend(history, "vs", gp.get_vs().data.cpu().numpy())

        logging.info(
            "epoch %d - tra_mse_val: %f - train_mse_out: %f"
            % (epoch, rv_eval["mse_val"], rv_eval["mse_out"])
        )

        # log to wandb
        if opt['use_wandb']:
            wandb.log({
                "epoch": epoch,
                "mse_val": rv_eval["mse_val"],
                "mse_out": rv_eval["mse_out"],
                "gp_nll": rv_eval["gp_nll"],
                "recon_term": rv_back["recon_term"],
                "pen_term": rv_back["pen_term"],
                "loss": rv_back["loss"],
                "vars": rv_eval["vars"],
            })

        # callback?
        if epoch % opt['epoch_cb'] == 0:
            logging.info("epoch %d - executing callback" % epoch)
            ffile = os.path.join(opt['outdir'], "plot.%.5d.png" % epoch)
            callback_gppvae(epoch, history, covs, imgs, ffile)
            
            # log plot to wandb
            if opt['use_wandb']:
                wandb.log({"reconstructions": wandb.Image(ffile)})

    # finish wandb run
    if opt['use_wandb']:
        wandb.finish()


def encode_Y(vae, train_queue):

    vae.eval()

    with torch.no_grad():

        n = train_queue.dataset.Y.shape[0]
        Zm = Variable(torch.zeros(n, vae_cfg["zdim"]), requires_grad=False).cuda()
        Zs = Variable(torch.zeros(n, vae_cfg["zdim"]), requires_grad=False).cuda()

        for batch_i, data in enumerate(train_queue):
            y = data[0].cuda()
            idxs = data[-1].cuda()
            zm, zs = vae.encode(y)
            Zm[idxs], Zs[idxs] = zm.detach(), zs.detach()

    return Zm, Zs


def eval_step(vae, gp, vm, val_queue, Zm, Vt, Vv):

    rv = {}

    with torch.no_grad():

        _X = vm.x().data.cpu().numpy()
        _W = vm.v().data.cpu().numpy()
        covs = {"XX": sp.dot(_X, _X.T), "WW": sp.dot(_W, _W.T)}
        rv["vars"] = gp.get_vs().data.cpu().numpy()
        # out of sample
        vs = gp.get_vs()
        U, UBi, _ = gp.U_UBi_Shb([Vt], vs)
        Kiz = gp.solve(Zm, U, UBi, vs)
        Zo = vs[0] * Vv.mm(Vt.transpose(0, 1).mm(Kiz))
        mse_out = Variable(torch.zeros(Vv.shape[0], 1), requires_grad=False).cuda()
        mse_val = Variable(torch.zeros(Vv.shape[0], 1), requires_grad=False).cuda()
        for batch_i, data in enumerate(val_queue):
            idxs = data[-1].cuda()
            Yv = data[0].cuda()
            Zv = vae.encode(Yv)[0].detach()
            Yr = vae.decode(Zv)
            Yo = vae.decode(Zo[idxs])
            mse_out[idxs] = (
                ((Yv - Yo) ** 2).view(Yv.shape[0], -1).mean(1)[:, None].detach()
            )
            mse_val[idxs] = (
                ((Yv - Yr) ** 2).view(Yv.shape[0], -1).mean(1)[:, None].detach()
            )
            # store a few examples
            if batch_i == 0:
                imgs = {}
                imgs["Yv"] = Yv[:24].data.cpu().numpy().transpose(0, 2, 3, 1)
                imgs["Yr"] = Yr[:24].data.cpu().numpy().transpose(0, 2, 3, 1)
                imgs["Yo"] = Yo[:24].data.cpu().numpy().transpose(0, 2, 3, 1)
        rv["mse_out"] = float(mse_out.data.mean().cpu())
        rv["mse_val"] = float(mse_val.data.mean().cpu())

    return rv, imgs, covs


def backprop_and_update(
    vae, gp, vm, train_queue, Dt, Wt, Eps, Zb, Vbs, vbs, vae_optimizer, gp_optimizer
):

    rv = {}

    vae_optimizer.zero_grad()
    gp_optimizer.zero_grad()
    vae.train()
    gp.train()
    vm.train()
    for batch_i, data in enumerate(train_queue):

        # subset data
        y = data[0].cuda()
        eps = Eps[data[-1]]
        _d = Dt[data[-1]]
        _w = Wt[data[-1]]
        _Zb = Zb[data[-1]]
        _Vbs = [Vbs[0][data[-1]]]

        # forward vae
        zm, zs = vae.encode(y)
        z = zm + zs * eps
        yr = vae.decode(z)
        recon_term, mse = vae.nll(y, yr)

        # forward gp
        _Vs = [vm(_d, _w)]
        gp_nll_fo = gp.taylor_expansion(z, _Vs, _Zb, _Vbs, vbs) / vae.K

        # penalization
        pen_term = -0.5 * zs.sum(1)[:, None] / vae.K

        # loss and backward
        loss = (recon_term + gp_nll_fo + pen_term).sum()
        loss.backward()

        # store stuff
        _n = train_queue.dataset.Y.shape[0]
        smartSum(rv, "mse", float(mse.data.sum().cpu()) / _n)
        smartSum(rv, "recon_term", float(recon_term.data.sum().cpu()) / _n)
        smartSum(rv, "pen_term", float(pen_term.data.sum().cpu()) / _n)

    vae_optimizer.step()
    gp_optimizer.step()

    return rv


if __name__ == "__main__":
    main()
