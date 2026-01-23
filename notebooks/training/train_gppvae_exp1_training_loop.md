# Training Loop with Per-View Metrics (Cell 13)

```python
import time
from IPython.display import clear_output

history = {}
start_time = time.time()

print(f"🚀 Starting GP-VAE Experiment #1 training for {CONFIG['epochs']} epochs...")
print("=" * 80)
print("Training mode: JOINT OPTIMIZATION (VAE + GP updated together)")
print(f"Experiment: Hard Held-Out Views")
print(f"  Training views: {CONFIG.get('train_view_indices', 'all')}")
print(f"  Validation views: {CONFIG.get('val_view_indices', 'all')}")
print("=" * 80)

for epoch in range(CONFIG['epochs']):
    epoch_start = time.time()

    # 1. Encode all training images
    Zm, Zs = encode_Y(vae, train_queue)

    # 2. Sample latent codes
    Eps = Variable(torch.randn(*Zs.shape), requires_grad=False).cuda()
    Z = Zm + Eps * Zs

    # 3. Compute variance matrices
    Vt = vm(Dt, Wt).detach()
    Vv = vm(Dv, Wv).detach()

    # 4. Evaluate on validation set (with per-view metrics)
    rv_eval, imgs, covs = eval_step(vae, gp, vm, val_queue, Zm, Vt, Vv, Wv)

    # 5. Compute GP Taylor expansion coefficients
    Zb, Vbs, vbs, gp_nll = gp.taylor_coeff(Z, [Vt])
    rv_eval["gp_nll"] = float(gp_nll.data.mean().cpu()) / vae.K

    # 6. Joint training step (VAE + GP)
    rv_back = backprop_and_update(
        vae, gp, vm, train_queue, Dt, Wt, Eps,
        Zb, Vbs, vbs, vae_optimizer, gp_optimizer
    )
    rv_back["loss"] = rv_back["recon_term"] + rv_eval["gp_nll"] + rv_back["pen_term"]

    # Store history
    smartAppendDict(history, rv_eval)
    smartAppendDict(history, rv_back)
    smartAppend(history, "vs", gp.get_vs().data.cpu().numpy())

    epoch_time = time.time() - epoch_start
    total_time = time.time() - start_time

    # 🔬 Compute diagnostic metrics
    train_val_gap = rv_back["mse"] - rv_eval["mse_val"]
    val_out_gap = rv_eval["mse_out"] - rv_eval["mse_val"]

    vs = gp.get_vs().data.cpu().numpy()
    variance_ratio = vs[0] / (vs[0] + vs[1])

    # Check if kernel has learnable lengthscale
    learned_lengthscale = None
    if hasattr(vm, 'view_kernel') and hasattr(vm.view_kernel, 'log_lengthscale'):
        learned_lengthscale = torch.exp(vm.view_kernel.log_lengthscale).item()

    # Print progress
    if epoch % 5 == 0 or epoch == CONFIG['epochs'] - 1:
        print(f"Epoch {epoch:4d}/{CONFIG['epochs']} | "
              f"MSE train: {rv_back['mse']:.6f} | "
              f"MSE val: {rv_eval['mse_val']:.6f} | "
              f"MSE out: {rv_eval['mse_out']:.6f} | "
              f"GP NLL: {rv_eval['gp_nll']:.4f} | "
              f"Gap(T-V): {train_val_gap:.6f} | "
              f"Gap(V-O): {val_out_gap:.6f} | "
              f"v₀/(v₀+v₁): {variance_ratio:.3f}" +
              (f" | ℓ: {learned_lengthscale:.3f}" if learned_lengthscale else "") +
              f" | Time: {epoch_time:.1f}s")
        
        # Print per-view breakdown (Experiment #1 specific)
        if CONFIG['view_split_mode'] == 'by_view' and epoch % 10 == 0:
            print("   Per-view MSE_out:")
            view_names = {0: "90L", 1: "60L", 2: "45L", 3: "30L", 4: "00F", 
                         5: "30R", 6: "45R", 7: "60R", 8: "90R"}
            for view_idx in sorted(rv_eval['mse_out_per_view'].keys()):
                mse = rv_eval['mse_out_per_view'][view_idx]
                view_name = view_names.get(view_idx, f"V{view_idx}")
                print(f"      {view_name}: {mse:.6f}")

    # Log to W&B
    if CONFIG['use_wandb']:
        log_dict = {
            "epoch": epoch,
            "mse_train": rv_back["mse"],
            "mse_val": rv_eval["mse_val"],
            "mse_out": rv_eval["mse_out"],
            "gp_nll": rv_eval["gp_nll"],
            "recon_term": rv_back["recon_term"],
            "pen_term": rv_back["pen_term"],
            "loss": rv_back["loss"],
            "vars": rv_eval["vars"],
            "time/epoch_seconds": epoch_time,
            # 🔬 Diagnostic metrics
            "diagnostics/gap_train_val": train_val_gap,
            "diagnostics/gap_val_out": val_out_gap,
            "diagnostics/variance_ratio": variance_ratio,
            "vars/v0_object": vs[0],
            "vars/v1_noise": vs[1],
        }

        # Add lengthscale if available
        if learned_lengthscale is not None:
            log_dict["kernel/lengthscale"] = learned_lengthscale
        
        # Add per-view metrics (Experiment #1 specific)
        if 'mse_val_per_view' in rv_eval:
            for view_idx, mse in rv_eval['mse_val_per_view'].items():
                view_names = {0: "90L", 1: "60L", 2: "45L", 3: "30L", 4: "00F",
                             5: "30R", 6: "45R", 7: "60R", 8: "90R"}
                view_name = view_names.get(view_idx, f"V{view_idx}")
                log_dict[f"mse_val_per_view/{view_name}"] = mse
        
        if 'mse_out_per_view' in rv_eval:
            for view_idx, mse in rv_eval['mse_out_per_view'].items():
                view_names = {0: "90L", 1: "60L", 2: "45L", 3: "30L", 4: "00F",
                             5: "30R", 6: "45R", 7: "60R", 8: "90R"}
                view_name = view_names.get(view_idx, f"V{view_idx}")
                log_dict[f"mse_out_per_view/{view_name}"] = mse

        wandb.log(log_dict)

    # Save checkpoint
    if epoch % CONFIG['epoch_cb'] == 0 or epoch == CONFIG['epochs'] - 1:
        logging.info(f"Epoch {epoch} - saving checkpoint")

        # Save VAE weights
        vae_file = os.path.join(wdir, f"vae_weights.{epoch:05d}.pt")
        torch.save(vae.state_dict(), vae_file)

        # Save GP weights
        gp_file = os.path.join(wdir, f"gp_weights.{epoch:05d}.pt")
        torch.save({
            'gp_state': gp.state_dict(),
            'vm_state': vm.state_dict(),
            'gp_params': gp_params.state_dict(),
        }, gp_file)

        # Save visualization
        ffile = os.path.join(fdir, f"plot.{epoch:05d}.png")
        callback_gppvae(epoch, history, covs, imgs, ffile)

        if CONFIG['use_wandb']:
            wandb.log({
                "reconstructions": wandb.Image(ffile),
                "covariances/XX": wandb.Image(ffile),
            })

        print(f"  ✓ Checkpoint saved at epoch {epoch}")

# At the end, enhanced summary with per-view breakdown
total_time = time.time() - start_time
print("\n" + "=" * 80)
print(f"✅ GP-VAE Experiment #1 training complete!")
print(f"   Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
print(f"   Average time per epoch: {total_time/CONFIG['epochs']:.1f} seconds")
print(f"   Final training MSE: {rv_back['mse']:.6f}")
print(f"   Final validation MSE: {rv_eval['mse_val']:.6f}")
print(f"   Final out-of-sample MSE: {rv_eval['mse_out']:.6f}")
print(f"   Final GP NLL: {rv_eval['gp_nll']:.4f}")

print(f"\n🔬 Final Diagnostics:")
print(f"   Train-Val Gap: {train_val_gap:.6f} (lower = less overfitting)")
print(f"   Val-Out Gap: {val_out_gap:.6f} (lower = better GP interpolation)")
print(f"   Variance Ratio: {variance_ratio:.3f} (higher = more structure learned)")
if learned_lengthscale is not None:
    print(f"   Learned Lengthscale: {learned_lengthscale:.3f}")

# Experiment #1 specific: Per-view breakdown
if CONFIG['view_split_mode'] == 'by_view' and 'mse_out_per_view' in rv_eval:
    print(f"\n📊 Final Per-View MSE_out (Experiment #1):")
    view_names = {0: "90L (-90°)", 1: "60L (-60°)", 2: "45L (-45°)", 3: "30L (-30°)", 
                 4: "00F (0°)", 5: "30R (+30°)", 6: "45R (+45°)", 7: "60R (+60°)", 8: "90R (+90°)"}
    
    # Separate training and validation views
    train_view_indices = CONFIG.get('train_view_indices', [])
    val_view_indices = CONFIG.get('val_view_indices', [])
    
    print("   VALIDATION VIEWS (held-out, extreme angles):")
    extreme_mses = []
    for view_idx in sorted(rv_eval['mse_out_per_view'].keys()):
        if view_idx in val_view_indices:
            mse = rv_eval['mse_out_per_view'][view_idx]
            extreme_mses.append(mse)
            view_name = view_names.get(view_idx, f"V{view_idx}")
            print(f"      {view_name:15s}: {mse:.6f}")
    
    if extreme_mses:
        avg_extreme = np.mean(extreme_mses)
        print(f"\n   Average MSE on extreme angles: {avg_extreme:.6f}")
        print(f"   Overall MSE_out: {rv_eval['mse_out']:.6f}")

if CONFIG['use_wandb']:
    wandb.finish()
    print("\n🔗 View detailed results in W&B dashboard")
```
