# backend/app/models/inpainting.py
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import logging
from typing import Tuple, List
import albumentations as A
from skimage import restoration, filters
from scipy import ndimage

logger = logging.getLogger(__name__)

class LaMaInpainting(nn.Module):
    """LaMa Inpainting model for perfect background reconstruction"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained LaMa model
        self.model = self._load_lama_model()
        
    def _load_lama_model(self):
        # Load from torch hub or local
        # Simplified - in production use actual LaMa weights
        return None
        
    @torch.no_grad()
    def inpaint(self, image, mask):
        """Advanced inpainting using multiple techniques"""
        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2,0,1).float() / 255.0
        
        # LaMa inpainting
        if self.model is not None:
            result = self.model(image.unsqueeze(0).to(self.device), 
                               mask.unsqueeze(0).to(self.device))
            result = result.squeeze(0).cpu().numpy()
            result = np.transpose(result, (1,2,0)) * 255
            result = result.astype(np.uint8)
        else:
            # Fallback to OpenCV inpainting
            result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
            
        return result

class InpaintingModel:
    def __init__(self, config):
        self.config = config
        self.lama = LaMaInpainting(config)
        
    async def remove_text_perfect(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Remove text with perfect background reconstruction
        """
        x1, y1, x2, y2 = bbox
        
        # Create mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        
        # Method 1: LaMa AI inpainting
        result = self.lama.inpaint(image, mask)
        
        # Method 2: Patch-based inpainting for texture preservation
        result = self._patch_based_inpaint(result, mask, image)
        
        # Method 3: Edge-aware inpainting for sharp edges
        result = self._edge_aware_inpaint(result, mask, image)
        
        # Method 4: Color harmonization
        result = self._color_harmonize(result, mask, image)
        
        return result
    
    def _patch_based_inpaint(self, image, mask, original):
        """Use similar patches from image to fill region"""
        result = image.copy()
        region = (mask > 0)
        
        if not np.any(region):
            return result
            
        # Find similar patches from outside mask
        patch_size = 15
        margin = 20
        
        # Extract region boundary
        dilated = cv2.dilate(mask, np.ones((5,5), np.uint8))
        boundary = dilated - mask
        
        # Sample patches from boundary
        boundary_points = np.argwhere(boundary > 0)
        
        if len(boundary_points) > 0:
            # Randomly sample patches and blend
            for _ in range(min(50, len(boundary_points))):
                y, x = boundary_points[np.random.randint(len(boundary_points))]
                
                patch_y = max(0, min(image.shape[0]-patch_size, y - patch_size//2))
                patch_x = max(0, min(image.shape[1]-patch_size, x - patch_size//2))
                
                patch = original[patch_y:patch_y+patch_size, 
                                patch_x:patch_x+patch_size]
                
                # Find best matching position inside mask using NCC
                best_pos = self._find_best_match(image, patch, region)
                if best_pos:
                    py, px = best_pos
                    result[py:py+patch_size, px:px+patch_size] = patch
        
        return result
    
    def _edge_aware_inpaint(self, image, mask, original):
        """Preserve edges during inpainting"""
        # Detect edges in original
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges near mask
        edge_mask = cv2.dilate(edges, np.ones((3,3), np.uint8))
        edge_region = edge_mask * mask
        
        if np.any(edge_region):
            # Use telea inpainting for edges
            edge_inpainted = cv2.inpaint(original, edge_region, 3, cv2.INPAINT_TELEA)
            
            # Blend with result
            alpha = 0.7
            image = cv2.addWeighted(image, 1-alpha, edge_inpainted, alpha, 0)
        
        return image
    
    def _color_harmonize(self, image, mask, original):
        """Harmonize colors between inpainted region and original"""
        # Get color statistics from boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        dilated = cv2.dilate(mask, kernel)
        boundary = dilated - mask
        
        if np.any(boundary):
            # Extract boundary colors
            boundary_colors = original[boundary > 0]
            
            # Compute mean and std
            mean_color = np.mean(boundary_colors, axis=0)
            std_color = np.std(boundary_colors, axis=0) + 1e-6
            
            # Adjust inpainted region colors
            inpainted_region = image[mask > 0]
            inpainted_region = (inpainted_region - np.mean(inpainted_region, axis=0)) / (np.std(inpainted_region, axis=0) + 1e-6)
            inpainted_region = inpainted_region * std_color + mean_color
            
            # Clip to valid range
            inpainted_region = np.clip(inpainted_region, 0, 255).astype(np.uint8)
            
            # Put back
            image[mask > 0] = inpainted_region
        
        return image
    
    def _find_best_match(self, image, patch, region):
        """Find best position to place patch using NCC"""
        h, w = patch.shape[:2]
        best_score = -1
        best_pos = None
        
        # Only search in region
        y_idxs, x_idxs = np.where(region)
        if len(y_idxs) == 0:
            return None
            
        # Sample some positions
        sample_size = min(50, len(y_idxs))
        sample_indices = np.random.choice(len(y_idxs), sample_size, replace=False)
        
        for idx in sample_indices:
            y = y_idxs[idx]
            x = x_idxs[idx]
            
            if y + h > image.shape[0] or x + w > image.shape[1]:
                continue
                
            # Extract candidate region
            candidate = image[y:y+h, x:x+w]
            
            # Compute NCC
            score = self._compute_ncc(candidate, patch)
            
            if score > best_score:
                best_score = score
                best_pos = (y, x)
        
        return best_pos
    
    def _compute_ncc(self, patch1, patch2):
        """Normalized Cross Correlation"""
        patch1_flat = patch1.reshape(-1, 3).astype(np.float32)
        patch2_flat = patch2.reshape(-1, 3).astype(np.float32)
        
        # Normalize
        patch1_norm = (patch1_flat - np.mean(patch1_flat, axis=0)) / (np.std(patch1_flat, axis=0) + 1e-6)
        patch2_norm = (patch2_flat - np.mean(patch2_flat, axis=0)) / (np.std(patch2_flat, axis=0) + 1e-6)
        
        # Compute correlation
        corr = np.mean(patch1_norm * patch2_norm)
        
        return corr
