# backend/app/models/style_matcher.py
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import ImageFont, ImageDraw, Image
import logging
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
from torchvision import models, transforms

logger = logging.getLogger(__name__)

class FontStyleClassifier(nn.Module):
    """Deep learning model for font style classification"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class StyleMatcher:
    def __init__(self, config):
        self.config = config
        
        # Load font classification model
        self.font_classifier = self._load_font_classifier()
        
        # Load text effect detector
        self.effect_detector = self._load_effect_detector()
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def _load_font_classifier(self):
        """Load pre-trained font classifier"""
        # In production, load actual weights
        model = FontStyleClassifier()
        return model
        
    def _load_effect_detector(self):
        """Load text effect detection model"""
        return None
        
    async def analyze_text_style(self, text_region: np.ndarray) -> Dict:
        """
        Comprehensive text style analysis
        """
        style = {
            'font_family': 'Arial',  # default
            'font_weight': 'normal',
            'font_style': 'normal',
            'serif': False,
            'monospace': False,
            'handwriting': False,
            'confidence': 0.0
        }
        
        # Method 1: Deep learning classification
        if self.font_classifier is not None:
            tensor = self.transform(text_region).unsqueeze(0)
            with torch.no_grad():
                outputs = self.font_classifier(tensor)
                probs = torch.softmax(outputs, dim=1)
                
                # Map to style categories
                style = self._map_to_style(probs)
        
        # Method 2: Traditional feature analysis
        features = self._extract_style_features(text_region)
        
        # Combine results
        style.update(features)
        
        return style
    
    def _extract_style_features(self, region: np.ndarray) -> Dict:
        """Extract style features using traditional CV"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        
        # 1. Detect serifs using edge patterns
        edges = cv2.Canny(gray, 50, 150)
        
        # Horizontal edges at character tops might indicate serifs
        horizontal_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, 
                                           np.ones((1,5), np.uint8))
        serif_score = np.sum(horizontal_edges) / (edges.sum() + 1e-6)
        features['serif'] = serif_score > 0.3
        
        # 2. Detect monospace using character spacing
        # Find character boundaries
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        vertical_projection = np.sum(binary, axis=0)
        
        # Find gaps between characters
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i, val in enumerate(vertical_projection):
            if val == 0 and not in_gap:
                in_gap = True
                gap_start = i
            elif val > 0 and in_gap:
                in_gap = False
                gaps.append(i - gap_start)
        
        if len(gaps) > 1:
            gap_std = np.std(gaps)
            features['monospace'] = gap_std < 2  # Consistent gaps
        else:
            features['monospace'] = False
        
        # 3. Detect handwriting using stroke variability
        # Use HOG features for stroke analysis
        from skimage.feature import hog
        fd, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), visualize=True)
        
        # Handwriting tends to have more variable orientations
        orientation_std = np.std(fd)
        features['handwriting'] = orientation_std > 0.5
        
        return features
    
    async def detect_text_color(self, region: np.ndarray) -> List[int]:
        """
        Detect dominant text color (with shadow/outline handling)
        """
        # Reshape to list of pixels
        pixels = region.reshape(-1, 3)
        
        # Use KMeans to find dominant colors
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers and sizes
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        # Count pixels in each cluster
        counts = np.bincount(labels)
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        
        # First cluster is background, second is text
        if len(sorted_indices) >= 2:
            text_color = colors[sorted_indices[1]]
        else:
            text_color = colors[sorted_indices[0]]
        
        return text_color.tolist()
    
    async def detect_text_angle(self, region: np.ndarray) -> float:
        """
        Detect text rotation angle
        """
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        
        # Use Hough Line Transform to find dominant angle
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
        
        if lines is not None:
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = theta * 180 / np.pi - 90
                if abs(angle) < 45:  # Filter near-horizontal
                    angles.append(angle)
            
            if angles:
                return np.median(angles)
        
        return 0.0
    
    async def detect_text_effects(self, region: np.ndarray) -> Dict:
        """
        Detect text effects (shadow, outline, glow)
        """
        effects = {
            'has_shadow': False,
            'has_outline': False,
            'has_glow': False,
            'shadow_offset': (0, 0),
            'outline_color': None,
            'glow_color': None
        }
        
        # Detect shadow by analyzing edges
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        
        # Check for duplicate edges offset
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges
        dilated = cv2.dilate(edges, np.ones((3,3), np.uint8))
        
        # Find offset duplicates (shadow)
        for dx in [-2, -1, 1, 2]:
            for dy in [-2, -1, 1, 2]:
                shifted = np.roll(np.roll(edges, dx, axis=1), dy, axis=0)
                overlap = np.logical_and(dilated, shifted)
                
                if np.sum(overlap) > 0.3 * np.sum(edges):
                    effects['has_shadow'] = True
                    effects['shadow_offset'] = (dx, dy)
                    break
        
        return effects
    
    def _map_to_style(self, probs):
        """Map neural network outputs to style categories"""
        # Simplified mapping
        style_map = {
            0: {'font_family': 'Arial', 'serif': False, 'monospace': False},
            1: {'font_family': 'Times New Roman', 'serif': True, 'monospace': False},
            2: {'font_family': 'Courier New', 'serif': False, 'monospace': True},
            3: {'font_family': 'Impact', 'font_weight': 'bold', 'serif': False},
        }
        
        pred_class = torch.argmax(probs).item()
        return style_map.get(pred_class, style_map[0])
