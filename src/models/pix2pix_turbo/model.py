"""Vendored from img2img-turbo (GaParmar/img2img-turbo).

VAE encoder/decoder overrides with skip connections, and single-step scheduler.
"""

from diffusers import DDPMScheduler


def make_1step_sched(pretrained_model="stabilityai/sd-turbo"):
    """Create a DDPM scheduler configured for a single denoising step."""
    noise_scheduler_1step = DDPMScheduler.from_pretrained(
        pretrained_model, subfolder="scheduler"
    )
    noise_scheduler_1step.set_timesteps(1, device="cpu")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod
    return noise_scheduler_1step


def my_vae_encoder_fwd(self, sample):
    """Replacement VAE encoder forward that captures skip activations."""
    sample = self.conv_in(sample)
    l_blocks = []
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    """Replacement VAE decoder forward with skip connections from encoder."""
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [
            self.skip_conv_1,
            self.skip_conv_2,
            self.skip_conv_3,
            self.skip_conv_4,
        ]
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](
                self.incoming_skip_acts[::-1][idx] * self.gamma
            )
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample
