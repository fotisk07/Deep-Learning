import torch
from torch import nn

from clip_reproduction.models.text import TextTransformerCLIP
from clip_reproduction.models.vision import VisionTransformerCLIP


class CLIPModel(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        initial_logit_scale: float = 1 / 0.07,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(initial_logit_scale)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(token_ids)

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        image_features = self.encode_image(images)
        text_features = self.encode_text(token_ids)

        scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_image = scale * (image_features @ text_features.t())
        logits_per_text = logits_per_image.t()

        batch_size = images.size(0)
        targets = torch.arange(batch_size, device=images.device)
        loss_i = nn.functional.cross_entropy(logits_per_image, targets)
        loss_t = nn.functional.cross_entropy(logits_per_text, targets)
        loss = 0.5 * (loss_i + loss_t)

        return {
            "loss": loss,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
        }


def build_clip_model(
    image_size: int = 224,
    context_length: int = 77,
    embed_dim: int = 512,
    vision_width: int = 512,
    vision_layers: int = 8,
    vision_heads: int = 8,
    vision_patch_size: int = 16,
    text_width: int = 512,
    text_layers: int = 8,
    text_heads: int = 8,
    dropout: float = 0.0,
) -> CLIPModel:
    image_encoder = VisionTransformerCLIP(
        image_size=image_size,
        patch_size=vision_patch_size,
        width=vision_width,
        layers=vision_layers,
        heads=vision_heads,
        embed_dim=embed_dim,
        dropout=dropout,
    )
    text_encoder = TextTransformerCLIP(
        context_length=context_length,
        width=text_width,
        layers=text_layers,
        heads=text_heads,
        embed_dim=embed_dim,
        dropout=dropout,
    )
    return CLIPModel(image_encoder=image_encoder, text_encoder=text_encoder)
