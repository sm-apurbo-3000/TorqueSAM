import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

class MedSAMSegmenter:
    """MedSAM segmentation module that replaces QuickShift segmentation.
    Uses bounding box prompts from YOLOv8 to segment medical images."""
    
    def __init__(self, checkpoint_path, model_type='vit_b', device='cuda:0'):
        """
        Initialize MedSAM model.
        
        Args:
            checkpoint_path: Path to medsam_vit_b.pth checkpoint
            model_type: Model architecture type (default: 'vit_b')
            device: Device to run inference on
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Loading MedSAM model on {self.device}")
        
        # Load MedSAM model
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transform for preprocessing
        self.transform = ResizeLongestSide(1024)
        
        print("MedSAM model loaded successfully")
    
    def preprocess_image(self, image):
        """
        Preprocess image for MedSAM inference.

        Args:
            image: Input image as numpy array (H, W) or (H, W, C)

        Returns:
            Preprocessed image tensor
        """
        # Ensure image is uint8 in range [0, 255]
        if image.dtype != np.uint8:
            # Normalize to [0, 255] if needed
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        # Resize image (maintains aspect ratio)
        image_resized = self.transform.apply_image(image)
        resized_size = image_resized.shape[:2]

        # Convert to tensor and normalize to float
        image_tensor = torch.as_tensor(image_resized, device=self.device).float()
        image_tensor = image_tensor.permute(2, 0, 1).contiguous()[None, :, :, :]

        # Pad to 1024x1024 (required by SAM's positional embeddings)
        h, w = image_tensor.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        image_tensor = torch.nn.functional.pad(image_tensor, (0, padw, 0, padh))

        return image_tensor, resized_size
    
    def get_bbox_prompt(self, bbox, original_size):
        """
        Convert bounding box to MedSAM prompt format.
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            original_size: Original image size (H, W)
            
        Returns:
            Bounding box prompt tensor
        """
        # Ensure bbox is in correct format
        bbox_np = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
        
        # Transform bbox coordinates to match resized image
        bbox_transformed = self.transform.apply_boxes(bbox_np.reshape(1, 4), original_size)
        bbox_tensor = torch.as_tensor(bbox_transformed, dtype=torch.float, device=self.device)
        
        return bbox_tensor
    
    @torch.no_grad()
    def segment(self, image, bbox):
        """
        Perform segmentation using MedSAM with bounding box prompt.
        
        Args:
            image: Input image as numpy array (H, W) or (H, W, C)
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            Segmentation mask as numpy array (H, W) with integer labels
        """
        original_size = image.shape[:2]
        
        # Preprocess image
        image_tensor, resized_size = self.preprocess_image(image) #previous
        #image_tensor = torch.as_tensor(image_resized, device=self.device).float() / 255.0

        # Get image embedding (computed once per image)
        image_embedding = self.model.image_encoder(image_tensor)
        
        # Get bbox prompt
        bbox_prompt = self.get_bbox_prompt(bbox, original_size)
        
        # Generate mask using prompt encoder and mask decoder
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None,
            boxes=bbox_prompt,
            masks=None,
        )
        
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        # Upscale mask to original image size
        masks = torch.nn.functional.interpolate(
            low_res_masks,
            size=original_size,
            mode='bilinear',
            align_corners=False,
        )
        
        # Convert to binary mask
        mask = masks[0, 0].cpu().numpy()
        mask_binary = (mask > 0).astype(np.uint8)
        
        return mask_binary
    
    def segment_with_multiple_bboxes(self, image, bboxes):
        """
        Segment image with multiple bounding boxes.
        Each bbox gets a unique label in the output.
        
        Args:
            image: Input image as numpy array
            bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            
        Returns:
            Labeled segmentation mask where each region has a unique integer label
        """
        if len(bboxes) == 0:
            return np.zeros(image.shape[:2], dtype=np.int32)
        
        labeled_mask = np.zeros(image.shape[:2], dtype=np.int32)
        
        for idx, bbox in enumerate(bboxes, start=1):
            mask = self.segment(image, bbox)
            # Assign unique label to this region
            labeled_mask[mask > 0] = idx
        
        return labeled_mask


def medsam_segmentation(image, bboxes, medsam_model):
    """
    Main segmentation function that replaces quickshift segmentation.
    This function signature matches the existing preprocessing pipeline.
    
    Args:
        image: Input image (numpy array)
        bboxes: Bounding boxes from YOLOv8 detection
        medsam_model: Initialized MedSAMSegmenter instance
        
    Returns:
        Labeled segments (numpy array with integer labels)
    """
    if medsam_model is None:
        raise ValueError("MedSAM model not initialized")
    
    # Segment with all bounding boxes
    labeled_segments = medsam_model.segment_with_multiple_bboxes(image, bboxes)
    
    return labeled_segments


##################################################
# import numpy as np
# import torch
# import cv2
# from segment_anything import sam_model_registry
# from segment_anything.utils.transforms import ResizeLongestSide


# class MedSAMSegmenter:
#     """MedSAM segmentation module (SAM backbone) using bbox prompts."""

#     def __init__(self, checkpoint_path, model_type='vit_b', device='cuda:0'):
#         self.device = device if torch.cuda.is_available() else 'cpu'
#         print(f"Loading MedSAM model on {self.device}")

#         self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
#         self.model = self.model.to(self.device)
#         self.model.eval()

#         self.transform = ResizeLongestSide(1024)
#         print("MedSAM model loaded successfully")

#     def preprocess_image(self, image):
#         """
#         Returns:
#             image_tensor: float32 tensor on device, shape [1,3,1024,1024]
#             original_size: (H,W) of original input
#         """
#         if image is None or image.size == 0:
#             raise ValueError("Empty image provided to preprocess_image")

#         # Ensure uint8 [0,255]
#         if image.dtype != np.uint8:
#             if image.max() <= 1.0:
#                 image = (image * 255).astype(np.uint8)
#             else:
#                 image = image.astype(np.uint8)

#         # Ensure RGB
#         if image.ndim == 2:
#             image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#         elif image.ndim == 3 and image.shape[2] == 1:
#             image = np.repeat(image, 3, axis=2)

#         original_size = image.shape[:2]

#         # Resize longest side to 1024
#         image_resized = self.transform.apply_image(image)

#         # ✅ FIX: convert to float and normalize to [0,1]
#         image_tensor = torch.as_tensor(image_resized, device=self.device).float() / 255.0
#         image_tensor = image_tensor.permute(2, 0, 1).contiguous()[None, :, :, :]  # [1,3,H,W]

#         # Pad to 1024x1024
#         h, w = image_tensor.shape[-2:]
#         padh = 1024 - h
#         padw = 1024 - w
#         if padh < 0 or padw < 0:
#             # Shouldn't happen due to ResizeLongestSide(1024), but keep safe
#             image_tensor = torch.nn.functional.interpolate(
#                 image_tensor, size=(1024, 1024), mode="bilinear", align_corners=False
#             )
#         else:
#             image_tensor = torch.nn.functional.pad(image_tensor, (0, padw, 0, padh))

#         return image_tensor, original_size

#     def get_bbox_prompt(self, bbox, original_size):
#         """
#         bbox: [x1,y1,x2,y2] in original image coordinates
#         original_size: (H,W)
#         """
#         bbox_np = np.array([bbox[0], bbox[1], bbox[2], bbox[3]], dtype=np.float32)
#         bbox_transformed = self.transform.apply_boxes(bbox_np.reshape(1, 4), original_size)
#         bbox_tensor = torch.as_tensor(bbox_transformed, dtype=torch.float, device=self.device)
#         return bbox_tensor

#     @torch.no_grad()
#     def segment_with_multiple_bboxes(self, image, bboxes):
#         """
#         Faster: computes image embedding once, then decodes for each bbox.
#         Returns labeled mask (int32) where each bbox region gets unique label 1..N
#         """
#         if bboxes is None or len(bboxes) == 0:
#             return np.zeros(image.shape[:2], dtype=np.int32)

#         image_tensor, original_size = self.preprocess_image(image)

#         # Compute embedding once
#         image_embedding = self.model.image_encoder(image_tensor)

#         labeled_mask = np.zeros(original_size, dtype=np.int32)

#         for idx, bbox in enumerate(bboxes, start=1):
#             bbox_prompt = self.get_bbox_prompt(bbox, original_size)

#             sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
#                 points=None,
#                 boxes=bbox_prompt,
#                 masks=None,
#             )

#             low_res_masks, _ = self.model.mask_decoder(
#                 image_embeddings=image_embedding,
#                 image_pe=self.model.prompt_encoder.get_dense_pe(),
#                 sparse_prompt_embeddings=sparse_embeddings,
#                 dense_prompt_embeddings=dense_embeddings,
#                 multimask_output=False,
#             )

#             masks = torch.nn.functional.interpolate(
#                 low_res_masks,
#                 size=original_size,
#                 mode='bilinear',
#                 align_corners=False,
#             )

#             mask = masks[0, 0].detach().cpu().numpy()

#             # ✅ Better threshold than >0
#             mask_binary = (mask > 0.5).astype(np.uint8)

#             labeled_mask[mask_binary > 0] = idx

#         return labeled_mask


# def medsam_segmentation(image, bboxes, medsam_model):
#     """
#     Wrapper to match your preprocessing pipeline signature.
#     """
#     if medsam_model is None:
#         raise ValueError("MedSAM model not initialized")

#     return medsam_model.segment_with_multiple_bboxes(image, bboxes)
