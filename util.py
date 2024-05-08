import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import average_precision_score


def load_batch_to_device(batch, device):
    if type(batch)==torch.Tensor:
        batch = batch.to(device, non_blocking = True) 
        return
        
    for k in batch:
        if type(batch[k])==list and len(batch[k])==0: # no prefix
            continue
        elif type(batch[k])==torch.Tensor:
            batch[k] = batch[k].to(device, non_blocking = True) 
        elif type(batch[k][0])==torch.Tensor:
            batch[k] = [tensor_.to(device, non_blocking = True) for tensor_ in batch[k]]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def _calculate_ret_metric(img_logits, labels, mod_idxs, converted, device, chk_converted=False):
    img_logits = img_logits # B, 512
    class_labels = labels # list
    picture_emb = img_logits[mod_idxs==0, :]
    sketch_emb = converted[:, 0, 0, :][mod_idxs==1, :] if chk_converted else img_logits[mod_idxs==1, :] # sketch->photo
    picture_cls = torch.tensor(class_labels)[mod_idxs==0]
    sketch_cls = torch.tensor(class_labels)[mod_idxs==1]
    
    str_sim = torch.zeros(sketch_emb.size(0), picture_emb.size(0))
    sim_euc = torch.zeros(sketch_emb.size(0), picture_emb.size(0))

    MX = 512
    sketch_emb = sketch_emb.to(device)
    picture_emb = picture_emb.to(device)
    sketch_emb_norm = sketch_emb / sketch_emb.norm(dim=1)[:, None]
    picture_emb_norm = picture_emb / picture_emb.norm(dim=1)[:, None]
    sim_euc_list= []
    for i in range(sketch_cls.size(0)//MX + 1):
        sketch_emb_norm_batch = sketch_emb_norm[i*MX:(i+1)*MX]
        sim_euc_list.append(torch.mm(sketch_emb_norm_batch, picture_emb_norm.T))
    del sketch_emb_norm, sketch_emb, picture_emb, picture_emb_norm
    sim_euc = torch.cat(sim_euc_list, dim=0); del sim_euc_list
    
    str_sim_list = []
    for i in range(sketch_cls.size(0)//MX + 1):
        sketch_cls_batch = sketch_cls[i*MX:(i+1)*MX]
        str_sim_list.append(((sketch_cls_batch.unsqueeze(1) == picture_cls.unsqueeze(0)) * 1))
    del sketch_cls, picture_cls
    str_sim = torch.cat(str_sim_list, dim=0); del str_sim_list
    str_sim = str_sim.to(device)
    
    map_all, map_200, precision_100, precision_200, topk_photos = calculate(np.array(-sim_euc.cpu()), np.array(str_sim.cpu()), True)
    report_dict = {"mAP@all": map_all, "p@100": precision_100, "mAP@200": map_200,"p@200": precision_200,
                   "top10_retrieved": topk_photos}
    
    return report_dict

def calculate_ret_metric(img_logits, labels, mod_idxs, converted, device, log):
    
    log.info("[1/2] Original logits calculating")
    ori_output = _calculate_ret_metric(img_logits, labels, mod_idxs, converted, device, chk_converted=False)
    log.info("[1/2] Original logits calculation done")
    
    log.info("[2/2] Converted logits calculating")
    converted_output = _calculate_ret_metric(img_logits, labels, mod_idxs, converted, device, chk_converted=True)
    log.info("[2/2] Converted logits calculation done")

    res = {}
    for k in ori_output.keys():
        res[k] = [ori_output[k], converted_output[k]]
    
    return res



"""
copied & modified from https://github.com/buptLinfy/ZSE-SBIR/blob/main/utils/ap.py
"""

import time
import numpy as np
import torch
import multiprocessing
from joblib import delayed, Parallel


def calculate(distance, class_same, return_all=False, return_topk=False, out_path=None):
    arg_sort_topk = None
    
    if return_all:
        arg_sort_sim = distance.argsort()   # 得到从小到大索引值
        arg_sort_topk = arg_sort_sim[:, :10]
        sort_label = []
        for index in range(0, arg_sort_sim.shape[0]):
            # 将label重新排序，根据距离的远近，距离越近的排在前面
            sort_label.append(class_same[index, arg_sort_sim[index, :]])
        sort_label = np.array(sort_label)
    else: # quick-draw
        MX=2048
        sort_label = np.zeros_like(distance, dtype=np.int8)

        start_idx_ = 0
        for start_idx in range(0, len(distance), MX):
            end_idx = min(start_idx+MX, len(distance))
            distance_sorted_batch = distance[start_idx:end_idx].argsort() # MX, # of pictures
            if return_topk: 
                torch.save(distance_sorted_batch[:,:10], f"{out_path}/topk_idx_{start_idx_}")
                start_idx_ += 1
            class_same_batch = class_same[start_idx:end_idx] #(92291, 54358)
            sort_label[start_idx:end_idx] = np.take_along_axis(class_same_batch, distance_sorted_batch, axis=1)
            print(f"\t{start_idx} / {len(distance)}")

    # 多进程计算
    num_cores = min(multiprocessing.cpu_count(), 4)

    start = time.time()
    if return_all:
        aps_all = Parallel(n_jobs=num_cores)(
            delayed(voc_eval)(sort_label[iq]) for iq in range(distance.shape[0]))
        aps_200 = Parallel(n_jobs=num_cores)(
            delayed(voc_eval)(sort_label[iq], 200) for iq in range(distance.shape[0]))
        map_all = np.nanmean(aps_all)
        map_200 = np.nanmean(aps_200)

        precision_100 = Parallel(n_jobs=num_cores)(
            delayed(precision_eval)(sort_label[iq], 100) for iq in range(sort_label.shape[0]))
        precision_100 = np.nanmean(precision_100)
        precision_200 = Parallel(n_jobs=num_cores)(
            delayed(precision_eval)(sort_label[iq], 200) for iq in range(sort_label.shape[0]))
        precision_200 = np.nanmean(precision_200)
        
        # print("eval time:", time.time() - start)
    
    else: # quick-draw
        map_all, map_200, precision_100, precision_200 = None, None, None, None

        aps_all = Parallel(n_jobs=num_cores)(
            delayed(voc_eval)(sort_label[iq]) for iq in range(distance.shape[0]))
        map_all = np.nanmean(aps_all)
        print(f"aps_all done: {time.time() - start}")
        start = time.time()

        precision_200 = Parallel(n_jobs=num_cores)(
            delayed(precision_eval)(sort_label[iq], 200) for iq in range(sort_label.shape[0]))
        precision_200 = np.nanmean(precision_200)
        print(f"prec_200 done: {time.time() - start}")

    return map_all, map_200, precision_100, precision_200, arg_sort_topk



def voc_eval(sort_class_same, top=None):
    tp = sort_class_same
    tot_pos = np.sum(tp)
    fp = np.logical_not(tp)
    tot = tp.shape[0]
    if top is not None:
        top = min(top, tot)
        tp = tp[:top]
        fp = fp[:top]
        tot_pos = min(top, tot_pos)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    try:
        rec = tp / tot_pos
        precision = tp / (tp + fp)
    except:
        print("error", tot_pos)
        return np.nan

    ap = voc_ap(rec, precision)

    return ap


def precision_eval(sort_class_same, top=None):
    tp = sort_class_same
    tot_pos = np.sum(tp)

    if top is not None:
        top = min(top, tot_pos)
    else:
        top = tot_pos

    return np.mean(sort_class_same[:top])


def other_map1(sort_class_same, top=None):
    tp = sort_class_same
    tot_pos = np.sum(tp)
    tot = tp.shape[0]

    if top is not None:
        top = min(top, tot)
        tp = tp[:top]
        tot_pos = min(top, tot_pos)

    ap_sum = 0
    number = 0
    for i in range(len(tp)):
        if tp[i]:
            number += 1
            ap_sum += number / (i + 1)
            if number == tot_pos:
                break

    ap = ap_sum / (tot_pos + 1e-5)
    return ap


def voc_ap(rec, prec):
    mrec = np.append(0, rec)
    mrec = np.append(mrec, 1)

    mpre = np.append(0, prec)
    mpre = np.append(mpre, 0)

    for ii in range(len(mpre) - 2, -1, -1):
        mpre[ii] = max(mpre[ii], mpre[ii + 1])

    msk = [i != j for i, j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk] - mrec[0:-1][msk]) * mpre[1:][msk])
    return ap