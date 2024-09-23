import os
import sys
import numpy as np
import torch
import yaml
import glob
import argparse
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path
import cv2

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.insert(0, str(Path(__file__).resolve().parent / "lama"))
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo

from segment_anything import sam_model_registry
from segment_anything import SamPredictor
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download


SAM_MODELS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}

CACHE_PATH = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints"))


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print(f"Model loaded from {cache_file} \n => {log}")
    model.eval()
    return model


def transform_image(image) -> torch.Tensor:
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_transformed, _ = transform(image, None)
    return image_transformed


class ImageEditor:
    def __init__(self, lama_config, lama_ckpt, sam_type="vit_h", mask_dilate_factor: int = 30, return_prompts: bool = False, device: str = "cuda"):
        self.sam_type = sam_type
        self.return_prompts = return_prompts
        self.mask_dilate_factor = mask_dilate_factor
        self.device = torch.device(device)
        self._initialize_groundingdino()
        self._initialize_sam()
        self._initialize_lama(lama_config, lama_ckpt)

    def _initialize_sam(self):
        if self.sam_type is None:
            print("SAM type not specified. Defaulting to 'vit_h'.")
            self.sam_type = "vit_h"
        
        checkpoint_url = SAM_MODELS.get(self.sam_type, None)
        if checkpoint_url is None:
            raise ValueError(f"Invalid SAM type: {self.sam_type}. Please choose a valid SAM model type.")
        
        try:
            sam_model = sam_model_registry[self.sam_type]()
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
            sam_model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise ValueError(f"Error loading SAM model from URL: {checkpoint_url}. Ensure the model type and URL are correct. Error: {str(e)}")
        
        sam_model.to(device=self.device)
        self.sam = SamPredictor(sam_model)

    def _initialize_groundingdino(self):
        repo_id = "ShilongLiu/GroundingDINO"
        checkpoint_filename = "groundingdino_swinb_cogcoor.pth"
        config_filename = "GroundingDINO_SwinB.cfg.py"
        self.groundingdino = load_model_hf(repo_id, checkpoint_filename, config_filename)

    def _initialize_lama(self, lama_config, lama_ckpt):
        self.predict_config = OmegaConf.load(lama_config)
        self.predict_config.model.path = lama_ckpt

        train_config_path = os.path.join(
            self.predict_config.model.path, 'config.yaml')

        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(
            self.predict_config.model.path, 'models',
            self.predict_config.model.checkpoint
        )
        self.lama_model = load_checkpoint(
            train_config, checkpoint_path, strict=False, map_location='cpu')
        self.lama_model.freeze()
        if not self.predict_config.get('refine', False):
            self.lama_model.to(self.device)

    def predict_dino(self, image_pil, text_prompt, box_threshold, text_threshold):
        image_transformed = transform_image(image_pil)
        boxes, logits, phrases = predict(
            model=self.groundingdino,
            image=image_transformed,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            remove_combined=self.return_prompts,
            device=self.device
        )
        W, H = image_pil.size
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        return boxes, logits, phrases

    def predict_sam(self, image, boxes):
        image_array = np.asarray(image)
        self.sam.set_image(image_array)
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes, image_array.shape[:2])
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.sam.device),
            multimask_output=False,
        )
        return masks.cpu()

    def predict(self, image_pil, text_prompt, box_threshold=0.3, text_threshold=0.25):
        boxes, logits, phrases = self.predict_dino(image_pil, text_prompt, box_threshold, text_threshold)
        masks = torch.tensor([])
        if len(boxes) > 0:
            masks = self.predict_sam(image_pil, boxes)
            masks = masks.squeeze(1)
        return masks, boxes, phrases, logits

    @torch.no_grad()
    def inpaint_img_with_lama(self, img, mask, mod=8):
        assert len(mask.shape) == 2
        if np.max(mask) == 1:
            mask = mask * 255
        img = torch.from_numpy(img).float().div(255.)
        mask = torch.from_numpy(mask).float()

        batch = {}
        batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
        batch['mask'] = mask[None, None]
        unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
        batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
        batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
        batch = move_to_device(batch, self.device)
        batch['mask'] = (batch['mask'] > 0) * 1

        batch = self.lama_model(batch)
        cur_res = batch[self.predict_config.out_key][0].permute(1, 2, 0)
        cur_res = cur_res.detach().cpu().numpy()

        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        return cur_res

    def dilate_mask(self, mask, dilate_factor=15):
        mask = mask.astype(np.uint8)
        mask = cv2.dilate(
            mask,
            np.ones((dilate_factor, dilate_factor), np.uint8),
            iterations=1
        )
        return mask

    def segment_anything(self, image_path, text_prompt, output_image_path):
        image = Image.open(image_path).convert("RGB")
        masks, _, _, _ = self.predict(image, text_prompt)
        masks = [mask.squeeze().cpu().numpy().astype(np.uint8) * 255 for mask in masks]
        # boxes = [box.squeeze().cpu().numpy() for box in boxes]

        masked_array = np.array(image)

        for mask in masks:
          binary_mask = (mask == 255).astype(np.uint8)
          # Create a red mask image
          red_mask = np.zeros_like(np.array(image))
          red_mask[..., 0] = 255  # Set red channel to 255
          red_mask[..., 1] = 0    # Set green channel to 0
          red_mask[..., 2] = 0    # Set blue channel to 0

          # Apply the red mask to the original image where the mask is 1
          masked_array = np.where(binary_mask[..., None], red_mask, masked_array)

        Image.fromarray(masked_array).save(output_image_path)

    def remove_anything(self, image_path, text_prompt, output_image_path):
        image = Image.open(image_path).convert("RGB")
        masks, _, _, _ = self.predict(image, text_prompt)
        masks = [mask.squeeze().cpu().numpy().astype(np.uint8) * 255 for mask in masks]
        # boxes = [box.squeeze().cpu().numpy() for box in boxes]

        masks = [self.dilate_mask(mask, self.mask_dilate_factor) for mask in masks]

        img_inpainted = np.array(image)
        for mask in masks:
            img_inpainted = self.inpaint_img_with_lama(img_inpainted, mask)
        
        Image.fromarray(img_inpainted.astype(np.uint8)).save(output_image_path)

    def change_location_anything(self, image_path, text_prompt, shift_x, shift_y, output_image_path):
        image = Image.open(image_path).convert("RGB")
        masks, boxes, _, _ = self.predict(image, text_prompt)
        masks = [mask.squeeze().cpu().numpy().astype(np.uint8) * 255 for mask in masks]
        boxes = [box.squeeze().cpu().numpy() for box in boxes]

        dilated_masks = [self.dilate_mask(mask, self.mask_dilate_factor) for mask in masks]

        img_inpainted = np.array(image)
        for mask in dilated_masks:
            img_inpainted = self.inpaint_img_with_lama(img_inpainted, mask)
        
        image_np = np.array(image)
        for mask, box in zip(masks, boxes):
            object_img = cv2.bitwise_and(image_np, image_np, mask=mask)
            x, y, w, h = int(box[0]), int(box[1]), int(box[2] - box[0]) , int(box[3] - box[1])


            moved_image = np.zeros_like(image_np)

            # Determine the new position
            new_x = x + shift_x
            new_y = y + shift_y

            # Ensure the new position is within image boundaries
            if new_x < 0:
                new_x = 0
            if new_y < 0:
                new_y = 0
            if new_x + w > image.size[1]:
                new_x = image.size[1] - w
            if new_y + h > image.size[0]:
                new_y = image.shape[0] - h

            moved_image[new_y:new_y+h, new_x:new_x+w] = object_img[y:y+h, x:x+w]

            mask_indices = np.where(moved_image > 0)

            img_inpainted[mask_indices] = moved_image[mask_indices]

        Image.fromarray(img_inpainted.astype(np.uint8)).save(output_image_path)