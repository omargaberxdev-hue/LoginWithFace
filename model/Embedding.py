import logging
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoModel

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


class ImageEmbedder:
    def __init__(self, model_id: str, device: str = "cpu", input_size: int = 112):
        self.device = torch.device(device)
        self.model = (
            AutoModel.from_pretrained(model_id, trust_remote_code=True)
            .to(self.device)
            .eval()
        )
        self.preprocess = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)

    def embed(self, image: Image.Image) -> np.ndarray:
        x = self._to_tensor(image)
        try:
            with torch.inference_mode():
                emb = self.model(x)
        except Exception as e:
            raise EmbeddingError("Model inference failed") from e
        return F.normalize(emb, p=2, dim=1).squeeze(0).cpu().numpy()