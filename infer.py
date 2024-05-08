import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import logging
from pathlib import Path
from model.modeling_sbir import ModalAwareModel
from tqdm import tqdm

from data.get_loader import get_loader
from collections import defaultdict

from util import load_batch_to_device

@hydra.main(version_base=None, config_path = "setting", config_name="config-infer")
def main(cfg: DictConfig):
    
    load_path = Path("./outputs", cfg.path)
    assert load_path.exists(), FileExistsError()
    log = logging.getLogger(__name__)
    
    # LOAD TRAIN CONFIG
    cfg_path = Path(load_path, ".hydra", "config.yaml")
    train_cfg = OmegaConf.load(cfg_path)
    cfg.model.backbone = train_cfg.model.backbone
    cfg.model.max_length = train_cfg.model.max_length
    
    # LOAD MODEL
    model = ModalAwareModel(train_cfg.model)
    ckpt_path = Path(load_path, "model", f"{cfg.mode}.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    if "model_state_dict" in ckpt: # our previous code repository
        results = model.load_state_dict(ckpt["model_state_dict"])
    elif "model" in ckpt: # our current code repository
        results = model.load_state_dict(ckpt["model"])
    
    log.info(f"Model loaded from {ckpt_path}")
    log.info(f"\tResult: {results}")
    
    # DATA LOADER
    test_loader = get_loader(cfg, [cfg.dataset.test_idx], [[], [], []], is_train=False, split="test")
    log.info(f"Data loaders done")
    log.info(f"\tNum of test loader: {len(test_loader)}")
    
    # INFER & SAVE OUTPUTS
    out_path = Path(load_path, "infer_output", cfg.mode, str(cfg.dataset.test_idx))
    Path.mkdir(out_path, parents=True)
    losses, (txt_logits, img_logits, converted_logits) = predict(model, test_loader)
    
    losses_str = "\t".join([f"{k}: {v:.4f}" for k, v in losses.items()])
    log.info(f"Test Loss: {losses_str}")
    torch.save(txt_logits, Path(out_path, "txt_logits.pt"))
    torch.save(img_logits, Path(out_path, "img_logits.pt"))
    torch.save(converted_logits, Path(out_path, "converted_logits.pt"))
    log.info(f"Logits saved: {Path(out_path)}")
    

@torch.no_grad()
def predict(model, loader):
    
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    losses = defaultdict(list)
    
    txt_logits_list, img_logits_list, converted_logits_list = [], [], []
    for batch in tqdm(loader):
        load_batch_to_device(batch, device)
        results = model(batch)
        
        # save losses
        for k, x in results['loss'].items():
            if isinstance(x, float):
                losses[k].append(x)
            elif type(x)==torch.Tensor:
                losses[k].append(x.mean().item())
        
        # save logits
        txt_logits_list.append(results['txt_logits'])
        img_logits_list.append(results['img_logits'])
        converted_logits_list.append(results['converted_logits']) # torch.Size([128, 2, 4, 512])
        
    for k, v in losses.items():
        losses[k] = sum(v)/len(v)
    
    txt_logits_tensor = torch.cat(txt_logits_list, dim=0)
    img_logits_tensor = torch.cat(img_logits_list, dim=0)
    converted_logits_tensor = torch.cat(converted_logits_list, dim=0)
    
    return losses, (txt_logits_tensor, img_logits_tensor, converted_logits_tensor)
    

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()