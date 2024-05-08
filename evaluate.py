import hydra
import torch
from omegaconf import DictConfig

import json
from pathlib import Path
import logging

import pandas as pd
from util import calculate_ret_metric

DATASET_IDX_DICT = {0: "tuberlin", 1: "quickdraw", 2: "sketchy"}

@hydra.main(version_base=None, config_path = "setting", config_name="config-eval")
def main(cfg: DictConfig):
    
    dataset_name = DATASET_IDX_DICT[cfg.dataset.test_idx]
    load_path = Path("./outputs", cfg.path, "infer_output", cfg.mode, str(cfg.dataset.test_idx))
    log = logging.getLogger(__name__)
    
    # load
    img_logits = torch.load(Path(load_path, "img_logits.pt"))
    converted_logits = torch.load(Path(load_path, "converted_logits.pt"))
    
    csv_file = pd.read_csv(Path("./data", dataset_name, f"test_{dataset_name}.csv"))
    labels = torch.tensor(csv_file.loc[:, "label"].values)
    mod_idxs = torch.tensor(csv_file.loc[:, "m1"].values)
    sketch_paths = csv_file.loc[csv_file["m1"] == 1, "img_path"].to_list()
    photo_paths = csv_file.loc[csv_file["m1"] == 0, "img_path"].to_list()
    log.info("Loaded logits and dataset annot files")

    # calculate metric & get top 10 retrieved
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    res = calculate_ret_metric(img_logits, labels, mod_idxs, converted_logits, device, log)

    
    # save
    out = {}
    for k, v in res.items():
        if k != "top10_retrieved": # metric
            out[k] = v
            log.info(f"{k}: {v}")
        else:
            top10_retrived_out = {}
            for sketch_idx, (top10_retrieved_ori, top10_retrieved_converted) in enumerate(zip(*v)):
                sketch_path = sketch_paths[sketch_idx]
                top10_retrieved_ori_paths = [photo_paths[idx] for idx in top10_retrieved_ori]
                top10_retrieved_converted_paths = [photo_paths[idx] for idx in top10_retrieved_converted]
                top10_retrived_out[sketch_path] = {
                    "original": top10_retrieved_ori_paths,
                    "converted": top10_retrieved_converted_paths
                }
            out["top10_retrived"] = top10_retrived_out
    
    with open(Path(load_path, "eval_result.json"), "w") as json_file:
        json.dump(out, json_file, indent=4)



    

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()