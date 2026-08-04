from collections.abc import Iterable
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn

from hicpg.prompts import PROMPTS, validate_prompt_library
from hicpg.scoring import HierarchicalScores, score_embeddings


class HiCPG(nn.Module):
    def __init__(
        self,
        model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        level_weights: tuple[float, float, float] = (0.2, 0.5, 0.3),
        temperature: float = 0.02,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        validate_prompt_library()
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
        tokenizer = open_clip.get_tokenizer(model_name)
        self.model = model.eval().to(device)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.level_weights = level_weights
        self.temperature = temperature
        self.device_name = device
        self.class_names = tuple(PROMPTS)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("text_embeddings", self._encode_prompt_library(), persistent=False)

    @torch.inference_mode()
    def _encode_prompt_library(self) -> Tensor:
        class_tensors: list[Tensor] = []
        for name in self.class_names:
            level_tensors: list[Tensor] = []
            for prompts in PROMPTS[name].levels():
                tokens = self.tokenizer(list(prompts)).to(self.device_name)
                level_tensors.append(self.model.encode_text(tokens))
            class_tensors.append(torch.stack(level_tensors))
        return torch.stack(class_tensors)

    @torch.inference_mode()
    def encode_images(self, images: Iterable[Image.Image]) -> Tensor:
        tensors = [self.preprocess(image.convert("RGB")) for image in images]
        if not tensors:
            raise ValueError("image batch is empty")
        return self.model.encode_image(torch.stack(tensors).to(self.device_name))

    @torch.inference_mode()
    def forward(self, images: Iterable[Image.Image]) -> HierarchicalScores:
        embeddings = self.encode_images(images)
        return score_embeddings(embeddings, self.text_embeddings, self.level_weights, self.temperature)

    @torch.inference_mode()
    def predict(self, images: Iterable[Image.Image]) -> list[dict[str, Any]]:
        result = self.forward(images)
        indices = result.class_scores.argmax(dim=-1)
        return [
            {
                "prediction": self.class_names[int(index)],
                "confidence": float(result.probabilities[row, index].cpu()),
                "scores": {name: float(result.class_scores[row, column].cpu()) for column, name in enumerate(self.class_names)},
                "probabilities": {name: float(result.probabilities[row, column].cpu()) for column, name in enumerate(self.class_names)},
            }
            for row, index in enumerate(indices)
        ]
