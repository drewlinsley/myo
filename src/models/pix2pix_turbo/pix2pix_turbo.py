"""Vendored from img2img-turbo (GaParmar/img2img-turbo).

Single-step pix2pix model built on SD-Turbo with LoRA fine-tuning and
VAE skip connections for paired image-to-image translation.
"""

import copy
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig

from .model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd


class Pix2Pix_Turbo(nn.Module):
    """Single-step diffusion model for paired image-to-image translation.

    Uses SD-Turbo backbone with LoRA adapters on both UNet and VAE,
    plus learned skip connections from VAE encoder to decoder.

    Args:
        pretrained_path: path to a saved .pkl checkpoint (optional)
        pretrained_model: HuggingFace model ID for the SD-Turbo backbone
        lora_rank_unet: LoRA rank for UNet adapters
        lora_rank_vae: LoRA rank for VAE adapters
    """

    def __init__(
        self,
        pretrained_path=None,
        pretrained_model="stabilityai/sd-turbo",
        lora_rank_unet=8,
        lora_rank_vae=4,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model, subfolder="text_encoder"
        )
        self.sched = make_1step_sched(pretrained_model)

        vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae")
        # Monkey-patch encoder/decoder to support skip connections
        vae.encoder.forward = my_vae_encoder_fwd.__get__(
            vae.encoder, vae.encoder.__class__
        )
        vae.decoder.forward = my_vae_decoder_fwd.__get__(
            vae.decoder, vae.decoder.__class__
        )
        # Add skip connection convolutions to the decoder
        vae.decoder.skip_conv_1 = nn.Conv2d(
            512, 512, kernel_size=1, stride=1, bias=False
        )
        vae.decoder.skip_conv_2 = nn.Conv2d(
            256, 512, kernel_size=1, stride=1, bias=False
        )
        vae.decoder.skip_conv_3 = nn.Conv2d(
            128, 512, kernel_size=1, stride=1, bias=False
        )
        vae.decoder.skip_conv_4 = nn.Conv2d(
            128, 256, kernel_size=1, stride=1, bias=False
        )
        vae.decoder.ignore_skip = False

        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model, subfolder="unet"
        )

        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            unet_lora_config = LoraConfig(
                r=sd["rank_unet"],
                init_lora_weights="gaussian",
                target_modules=sd["unet_lora_target_modules"],
            )
            vae_lora_config = LoraConfig(
                r=sd["rank_vae"],
                init_lora_weights="gaussian",
                target_modules=sd["vae_lora_target_modules"],
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)
            self.target_modules_vae = sd["vae_lora_target_modules"]
            self.target_modules_unet = sd["unet_lora_target_modules"]
        else:
            # Initialize skip convs to near-zero so they start as no-ops
            nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)

            target_modules_vae = [
                "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
            vae_lora_config = LoraConfig(
                r=lora_rank_vae,
                init_lora_weights="gaussian",
                target_modules=target_modules_vae,
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

            target_modules_unet = [
                "to_k", "to_q", "to_v", "to_out.0",
                "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj",
            ]
            unet_lora_config = LoraConfig(
                r=lora_rank_unet,
                init_lora_weights="gaussian",
                target_modules=target_modules_unet,
            )
            unet.add_adapter(unet_lora_config)

            self.target_modules_vae = target_modules_vae
            self.target_modules_unet = target_modules_unet

        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae
        self.unet = unet
        self.vae = vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999]).long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def forward(self, c_t, prompt_tokens=None, deterministic=True):
        """Forward pass: single-step image-to-image translation.

        Args:
            c_t: input image tensor (B, 3, H, W) in [-1, 1]
            prompt_tokens: tokenized text prompt (B, seq_len) as input_ids
            deterministic: if True, use deterministic (no noise) mode

        Returns:
            output image tensor (B, 3, H, W) in [-1, 1]
        """
        # Ensure scheduler tensors are on the correct device
        if self.sched.alphas_cumprod.device != c_t.device:
            self.sched.alphas_cumprod = self.sched.alphas_cumprod.to(c_t.device)

        caption_enc = self.text_encoder(prompt_tokens)[0]
        encoded_control = (
            self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        )
        model_pred = self.unet(
            encoded_control,
            self.timesteps.to(c_t.device),
            encoder_hidden_states=caption_enc,
        ).sample
        x_denoised = self.sched.step(
            model_pred, self.timesteps[0].item(), encoded_control, return_dict=True
        ).prev_sample
        x_denoised = x_denoised.to(model_pred.dtype)
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        output_image = self.vae.decode(
            x_denoised / self.vae.config.scaling_factor
        ).sample.clamp(-1, 1)
        return output_image

    def save_model(self, path):
        """Save only trainable parameters (LoRA + skip convs + conv_in)."""
        sd = {
            "unet_lora_target_modules": self.target_modules_unet,
            "vae_lora_target_modules": self.target_modules_vae,
            "rank_unet": self.lora_rank_unet,
            "rank_vae": self.lora_rank_vae,
            "state_dict_unet": {
                k: v
                for k, v in self.unet.state_dict().items()
                if "lora" in k or "conv_in" in k
            },
            "state_dict_vae": {
                k: v
                for k, v in self.vae.state_dict().items()
                if "lora" in k or "skip" in k
            },
        }
        torch.save(sd, path)
