import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, CosineEmbeddingLoss
from PIL import Image
from typing import Optional, Tuple, Any, Union
from transformers import CLIPModel, GPT2LMHeadModel, AutoConfig

def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)
            
class ModalAwareEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.clip = CLIPModel.from_pretrained(self.config.backbone, logit_scale_init_value=self.config.temperature)
        self.clip.logit_scale = nn.Parameter(torch.tensor(self.config.temperature)) # 1-> 2.7182, 1.6094-> 5, 2.6592(default) -> 14.2848, 3.2189 -> 25, 4.6052(ori) -> 100
        self.mode_embedding = nn.Embedding(num_embeddings=config.n_mode, embedding_dim=self.config.d_model)
        self.discriminator = nn.Linear(config.d_model, config.n_mode)        
    
    def forward(self, batch):
        device = batch.input_ids.device
        B = batch.input_ids.size(0)
        
        # clip loss
        clip_out = self.clip(
            input_ids = batch.input_ids,
            attention_mask=batch.attention_mask,
            pixel_values = batch.pixel_values,
            return_loss=True
        )
        clip_loss = clip_out.loss
        clip_text_embeds = clip_out.text_embeds # normalized, B x d_model
        clip_image_embeds = clip_out.image_embeds # normalized, B x d_model
        
        # get z
        E_s = (clip_text_embeds + clip_image_embeds)/2 # B * d_model
        E_s = F.normalize(E_s, p=2, dim=-1)
        
        # semantic centor loss
        mode_emb = self.mode_embedding(batch.mod_idx)
        common_semantic_embeds_image = clip_image_embeds-mode_emb[:,0,:] # subtract image modality vectors
        common_semantic_embeds_text = clip_text_embeds-mode_emb[:,1,:] # subtract text modality vectors
        common_semantic_embeds_image_norm = F.normalize(common_semantic_embeds_image, dim=-1) # s_txt
        common_semantic_embeds_text_norm = F.normalize(common_semantic_embeds_text, dim=-1) # s_img
        
        text_centor_loss = CosineEmbeddingLoss()(common_semantic_embeds_text_norm, E_s[torch.arange(B)], torch.ones_like(batch.label))
        image_centor_loss = CosineEmbeddingLoss()(common_semantic_embeds_image_norm, E_s[torch.arange(B)], torch.ones_like(batch.label))
        semantic_center_loss = text_centor_loss + image_centor_loss

        # orthogonal loss
        E_m = self.mode_embedding(torch.arange(end=self.config.n_mode, device=device)) # num_mode x d_model
        extended_E_s = E_s.repeat(E_m.size(0), 1)
        extended_E_m = torch.repeat_interleave(E_m, repeats=torch.tensor(E_s.size(0), device=device), dim=0)
        sim_mat = (extended_E_s*extended_E_m).abs()
        orthogonal_loss = sim_mat.sum()/(torch.count_nonzero(extended_E_s.sum(-1)) + 1e-6)

        # get converted vectors
        converted_from_text = common_semantic_embeds_text.unsqueeze(1).repeat(1, self.config.n_mode, 1) + self.mode_embedding(torch.arange(start=0, end=self.config.n_mode, device=device)) # B, n_mode, d_model
        converted_from_text = F.normalize(converted_from_text, dim=-1)
        converted_from_image = common_semantic_embeds_image.unsqueeze(1).repeat(1, self.config.n_mode, 1) + self.mode_embedding(torch.arange(start=0, end=self.config.n_mode, device=device)) # B, n_mode, d_model
        converted_from_image = F.normalize(converted_from_image, dim=-1)

        image_converted_from_text = converted_from_text[torch.arange(start=0, end=batch.mod_idx.size(0), device=device), batch.mod_idx[:, 0], :]
        text_converted_from_image = converted_from_image[torch.arange(start=0, end=batch.mod_idx.size(0), device=device), batch.mod_idx[:, 1], :]
        
        reconstruct_loss = CosineEmbeddingLoss()(image_converted_from_text, clip_image_embeds, torch.ones_like(batch.label)) + \
                            CosineEmbeddingLoss()(text_converted_from_image, clip_text_embeds, torch.ones_like(batch.label))
        
        
        # discriminator loss
        mod_labels = torch.arange(start=0, end=self.config.n_mode, device=device, requires_grad=False).unsqueeze(0).repeat(converted_from_image.size(0),1)
        mod_labels_image = torch.where(mod_labels==batch.mod_idx[:,0].unsqueeze(1), -1, mod_labels).reshape(-1)
        mod_labels_text = torch.where(mod_labels==batch.mod_idx[:,1].unsqueeze(1), -1, mod_labels).reshape(-1)
        
        disc_loss = CrossEntropyLoss(ignore_index=-1)(self.discriminator(converted_from_text.reshape(-1, self.config.d_model)), mod_labels_text) + \
                    CrossEntropyLoss(ignore_index=-1)(self.discriminator(converted_from_image.reshape(-1, self.config.d_model)), mod_labels_image)
        disc_loss /= 2
        
        losses = {
            'clip_loss': clip_loss, 
            'semantic_center_loss': semantic_center_loss, 
            'orthogonal_loss': orthogonal_loss, 
            'discriminator_loss': disc_loss,
            'reconstruct_loss': reconstruct_loss,
            }
        
        return {
            'loss': losses,
            'txt_logits': clip_text_embeds,
            'img_logits': clip_image_embeds,
            'converted_logits': torch.stack([converted_from_image, converted_from_text,], dim=1), # B * 2+1(num_src_mode) * n_modes * d_model
            }
        
class ModalAwareModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.encoder = ModalAwareEncoder(config)

    def forward(self, batch):
        
        enc_out = self.encoder(batch)
        
        losses = enc_out["loss"]
        
        weighted_loss = 0.0
        denominator = 0.0
        if self.config.clip_loss_ratio > 0:
            weighted_loss += self.config.clip_loss_ratio*losses['clip_loss']
            denominator += self.config.clip_loss_ratio
        if self.config.common_semantic_loss_ratio > 0:
            weighted_loss += self.config.common_semantic_loss_ratio*losses['semantic_center_loss']
            denominator += self.config.common_semantic_loss_ratio
        if self.config.ortho_loss_ratio > 0:
            weighted_loss += self.config.ortho_loss_ratio*losses['orthogonal_loss']
            denominator += self.config.ortho_loss_ratio
        if self.config.disc_loss_ratio > 0:
            weighted_loss += self.config.disc_loss_ratio*losses['discriminator_loss']
            denominator += self.config.disc_loss_ratio
        if self.config.rec_loss_ratio > 0:
            weighted_loss += self.config.rec_loss_ratio*losses['reconstruct_loss']
            denominator += self.config.rec_loss_ratio
        weighted_loss /= denominator
        losses["weighted_loss"] = weighted_loss
        
        return {
            'loss': losses,
            'txt_logits': enc_out["txt_logits"],
            'img_logits': enc_out["img_logits"],
            'converted_logits': enc_out["converted_logits"],
        }