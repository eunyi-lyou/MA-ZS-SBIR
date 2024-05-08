import os
import hydra
import torch
from torch.cuda.amp import GradScaler, autocast
from omegaconf import DictConfig, OmegaConf
import wandb
import random
import logging
from tqdm import tqdm
from transformers import AutoConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from collections import defaultdict

from model.modeling_sbir import ModalAwareModel
from data.get_loader import get_loader
from util import load_batch_to_device, count_parameters

ALL_CLS_LISTS = [ # class index in train split
    list(range(1, 221)), # tu_berlin: (1~220)
    [ 22,  38, 107,  59,  19,  27,  33, 101,  76,  26,   7,  37,  75, 58,  98,  28,  15,  90,  57,  24,  92,  74,  52,  96,  65,  94, 91,   2,  55,  30,  11,  56,  62,  54, 106,  77,  23,  21,  61, 78,  48,  89,  84,  69,  16, 102,   4,  13, 109,  31,  72,  40, 32,  95,  10,  12,   5,  50,  34,  25,  99,  82,  70,  67,   1, 3,  97, 104,  51,  44,  66,  87,  81, 100,  64,  39,  60,  45, 73,  86], # quick_draw
    list(range(1, 105)), # sketchy: (1~104)
]

@hydra.main(version_base=None, config_path = "setting", config_name="config-train")
def main(cfg: DictConfig):
    
    os.makedirs(f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/model/', exist_ok=True)
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # LOGGER
    log = logging.getLogger(__name__)
    log.info(f"Loggers done")
    log.info(f"\tWeight will be saved at {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    
    # DATA LOADER
    train_loader, val_loader, test_loader = get_data_loader(cfg)
    log.info(f"Data loaders done")
    log.info(f"\tNum of train loaders: {len(train_loader)}, valid: {len(val_loader)}, test: {len(test_loader)}")
    
    # MODEL
    model = ModalAwareModel(cfg.model)
    trainable_param = count_parameters(model)
    log.info(f"Model done")
    log.info(f"\tNum of parameters: {trainable_param:,}")
    model = model.to(device)
    
    # TRAIN
    optim, lr_scheduler = get_optimizer_and_scheduler(cfg, model, len(train_loader))
    scaler = GradScaler()
    best_epoch = -1
    best_valid_loss = 100000
    global_step = 0
    
    
    for epoch in tqdm(range(cfg.train.epochs)):
        
        # train
        model.train()
        for step_i, batch in enumerate(train_loader):
            
            results, update = train_one_step(cfg, model, batch, optim, lr_scheduler, step_i, device, scaler)
            
            if update:
                global_step += 1
            
            # wandb log
            if global_step % cfg.train.log_every_step == 0:
                losses = {k: loss.mean().detach().cpu().item() for k, loss in results.items()} # detach
                lr = lr_scheduler.get_last_lr()[0]
                
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "loss": losses["weighted_loss"],
                        "clip_loss": losses["clip_loss"],
                        "semantic_center_loss": losses["semantic_center_loss"],
                        "orthogonal_loss": losses["orthogonal_loss"],
                        "discriminator_loss": losses["discriminator_loss"],
                        "reconstruct_loss": losses["reconstruct_loss"],
                        "lr": lr
                    },
                    step = global_step
                )
                losses_str = "\t".join([f"{k}: {v:.4f}" for k, v in losses.items()])
                log.info(f"Loss after {str(global_step).zfill(5)}: {losses_str}")
        
        # valid
        if (epoch+1) % cfg.train.eval_every_epoch == 0:
            if cfg.dataset.cls_ratio == 1.0 and len(val_loader) == 0:
                losses = predict(model, test_loader, device)
            else:
                losses = predict(model, val_loader, device)
            
            wandb.log({"valid_loss": losses["weighted_loss"]})
            
            losses_str = "\t".join([f"{k}: {v:.4f}" for k, v in losses.items()])
            log.info(f"Valid Loss after epoch {epoch+1}: {losses_str}")
            
            # renew best epoch
            if losses["weighted_loss"] <= best_valid_loss:
                if best_epoch != -1: log.info(f"Best epoch updated: {best_epoch} -> {epoch + 1}")
                best_epoch = epoch + 1
                best_valid_loss = losses["weighted_loss"]
                
                checkpoint = { 
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'lr_sched': lr_scheduler
                }
                torch.save(checkpoint, f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/model/best.pth')

        if epoch == cfg.train.epochs - 1:
            checkpoint = { 
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'lr_sched': lr_scheduler
            }
            torch.save(checkpoint, f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/model/last.pth')
  
    return

def get_optimizer_and_scheduler(cfg, model, num_batches):
    clip_params = list(model.encoder.clip.parameters())
    base_params = [p for n, p in model.named_parameters() if 'clip' not in n]
    optim = AdamW(
        [
            {'params': base_params},
            {'params': clip_params, 'lr': cfg.train.clip_lr}
        ],
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay
        )
    
    t_total = num_batches // cfg.train.gradient_accumulation_steps * cfg.train.epochs
    warmup_iters = int(t_total * cfg.train.warmup_ratio)
    lr_scheduler = get_constant_schedule_with_warmup(optim, warmup_iters)
    # lr_scheduler = get_linear_schedule_with_warmup(optim, warmup_iters, t_total)
    
    return optim, lr_scheduler

def _get_random_classes(ds_idx, cls_ratio, seen_cls=None):
    """select unseen classes in given dataset

    Args:
        ds_idx (List[int]): list of dataset idx to train
        cls_ratio (float): class ratio to train with (only applicable in training)
        seen_cls (List[int], optional): already selected training classes

    Returns:
        List[int]: class indexes selected for dataset split
    """
    
    cls_lists = [None, None, None]

    for ds_i in ds_idx:
        cls_list = ALL_CLS_LISTS[ds_i]
        if seen_cls is None: # training -> select random classes
            selected_list = random.sample(cls_list, int(len(cls_list)*cls_ratio))
        else: # validating -> select remaining indexes
            selected_list = list(set(cls_list) - set(seen_cls[ds_i]))
        cls_lists[ds_i] = selected_list
        return cls_lists

def get_data_loader(cfg):
    train_loaders, val_loaders, test_loaders = [], [], []
    
    train_cls_list = _get_random_classes(cfg.dataset.train_idx, cfg.dataset.cls_ratio)
    train_loaders = get_loader(cfg, cfg.dataset.train_idx, train_cls_list, is_train=True, split='train')
    
    if cfg.dataset.cls_ratio == 1.0:
        val_loaders = []
    else:
        valid_cls_list = _get_random_classes(cfg.dataset.valid_idx, cfg.dataset.cls_ratio, train_cls_list)
        val_loaders = get_loader(cfg, cfg.dataset.valid_idx, valid_cls_list, is_train=False, split='val')
        
    test_loaders = get_loader(cfg, cfg.dataset.test_idx, [[], [], []], is_train=False, split="test")
    
    return train_loaders, val_loaders, test_loaders

def get_model_config(cfg):
    model_cfg = AutoConfig.from_pretrained(cfg.model.backbone)
    for k, v in cfg.model.items():
        model_cfg[k] = v
    return model_cfg

def train_one_step(cfg, model, batch, optim, scheduler, step, device, scaler=None):
    # set flag for gradient update
    update = False
    if (1+step) % cfg.train.gradient_accumulation_steps==0: 
        update = True
    
    # forward
    load_batch_to_device(batch, device)
    if cfg.use_mixed_precision:
        with autocast():
            results = model(batch)
    else:
        results = model(batch)
    
    # backward
    results = results['loss']
    if cfg.use_mixed_precision:
        scaler.scale(results['weighted_loss'].mean()/cfg.train.gradient_accumulation_steps).backward()
    else:
        (results['weighted_loss'].mean()/cfg.train.gradient_accumulation_steps).backward()
    
    # gradient update
    if update:
        if cfg.use_mixed_precision:
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()
            
        scheduler.step()
        
        optim.zero_grad(set_to_none=True)

    return results, update

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    losses = defaultdict(list)
    
    for batch in loader:
        load_batch_to_device(batch, device)
        results = model(batch)
        for k, x in results['loss'].items():
            if isinstance(x, float):
                losses[k].append(x)
            elif type(x)==torch.Tensor:
                losses[k].append(x.mean().item())
    
    for k, v in losses.items():
        losses[k] = sum(v)/len(v)
    
    return losses

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    wandb.init(project="sbir")
    main()