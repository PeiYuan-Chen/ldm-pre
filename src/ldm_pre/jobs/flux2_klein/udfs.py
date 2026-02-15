import numpy as np
import torch
from diffusers import AutoencoderKLFlux2
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
from diffusers import Flux2KleinPipeline


from ldm_pre.udfs import Transform, LatentEncoder
from ldm_pre.schema import Cols


class Flux2KleinTransform(Transform):
    def __init__(
        self,
        cols: Cols,
        target_height: int,
        target_width: int,
        *,
        pretrained_model_name_or_path: str = "black-forest-labs/FLUX.2-klein-base-9B",
        max_length: int = 512,
    ) -> None:
        super().__init__(cols, target_height, target_width)

        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        self.max_length = max_length

    def tokenize(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        messages = [{"role": "user", "content": text}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return (
            inputs["input_ids"][0],  # [L]
            inputs["attention_mask"][0],  # [L]
        )


class Flux2KleinLatentEncoder(LatentEncoder):
    def __init__(
        self,
        cols: Cols,
        *,
        pretrained_model_name_or_path: str = "black-forest-labs/FLUX.2-klein-base-9B",
        hidden_states_layers: tuple[int] = (9, 18, 27),
    ) -> None:
        super().__init__(cols)

        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        vae = AutoencoderKLFlux2.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
        )
        # flux vae is stable in bf16 so load it in weight_dtype to reduce memory
        self.vae = vae.requires_grad_(False).to(
            dtype=torch.bfloat16, device=self.device
        )
        self.latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
            self.device
        )
        self.latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(self.device)

        text_encoder = Qwen3ForCausalLM.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        )
        self.text_encoder = text_encoder.requires_grad_(False).to(
            dtype=torch.bfloat16, device=self.device
        )
        self.hidden_states_layers = hidden_states_layers

    def encode_images(self, images: np.ndarray) -> torch.Tensor:
        pixel_values = torch.from_numpy(images).to(self.device, dtype=self.vae.dtype)
        image_latents = self.vae.encode(pixel_values).latent_dist.mode()
        image_latents = Flux2KleinPipeline._patchify_latents(image_latents)
        image_latents = (image_latents - self.latents_bn_mean) / self.latents_bn_std
        return image_latents

    def encode_text(
        self, input_ids: np.ndarray, attention_mask: np.ndarray
    ) -> torch.Tensor:
        input_ids = torch.from_numpy(input_ids).to(self.device)
        attention_mask = torch.from_numpy(attention_mask).to(self.device)

        output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        # Only use outputs from intermediate layers and stack them
        out = torch.stack(
            [output.hidden_states[k] for k in self.hidden_states_layers], dim=1
        )
        out = out.to(dtype=self.text_encoder.dtype, device=self.device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, num_channels * hidden_dim
        )
        return prompt_embeds
