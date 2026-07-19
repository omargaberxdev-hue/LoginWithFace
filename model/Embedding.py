import torch
import numpy as np
import open_clip
from PIL import Image


class ImageEmbedding:
    model = None
    preprocess = None

    @staticmethod
    def load_model(device: str = "cpu"):
        ImageEmbedding.device = device
        ImageEmbedding.model, _, ImageEmbedding.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai'
        )
        ImageEmbedding.model.to(device)
        ImageEmbedding.model.eval()


    @staticmethod
    def embed_image(image: Image.Image) -> np.ndarray:
        if ImageEmbedding.model is None:
            raise RuntimeError("Call load_model() before embed_image().")
        try :
            image_tensor = ImageEmbedding.preprocess(image).unsqueeze(0)

            with torch.no_grad():
                embedding = ImageEmbedding.model.encode_image(image_tensor)
        except :
            raise EmbeddedError("Error happing while Embeded the Image")

        return embedding.squeeze(0).numpy()