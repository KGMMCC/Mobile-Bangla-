# backend/app/models/ocr_model.py
import cv2
import numpy as np
import torch
import easyocr
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import logging
from typing import List, Dict
import asyncio

logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self, config):
        self.config = config
        
        # Initialize multiple OCR engines for best accuracy
        self.easy_ocr = easyocr.Reader(['bn', 'en'], gpu=torch.cuda.is_available())
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
        # Transformer-based OCR for challenging cases
        self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        
    async def detect_text_advanced(self, image: np.ndarray) -> List[Dict]:
        """
        Advanced text detection using multiple techniques
        """
        text_blocks = []
        
        # Method 1: EasyOCR for general text
        easy_results = self.easy_ocr.readtext(image)
        
        # Method 2: PaddleOCR for curved/angled text
        paddle_results = self.paddle_ocr.ocr(image, cls=True)
        
        # Method 3: MSER (Maximally Stable Extremal Regions) for small text
        mser_results = self._detect_mser_text(image)
        
        # Method 4: Contour-based detection for large text
        contour_results = self._detect_contour_text(image)
        
        # Combine all results with confidence scoring
        all_results = self._combine_detections([
            easy_results, 
            paddle_results[0] if paddle_results else [],
            mser_results,
            contour_results
        ])
        
        # Post-process and filter
        for result in all_results:
            if result['confidence'] > 0.6:  # High confidence threshold
                text_blocks.append(result)
        
        return text_blocks
    
    def _detect_mser_text(self, image):
        """Detect text using MSER algorithm"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        text_regions = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            if w > 20 and h > 10:  # Minimum text size
                text_region = image[y:y+h, x:x+w]
                text_regions.append({
                    'bbox': [x, y, x+w, y+h],
                    'confidence': 0.7,
                    'detector': 'mser'
                })
        
        return text_regions
    
    def _detect_contour_text(self, image):
        """Detect text using contour analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 30 and h > 15 and w/h < 10:  # Text-like aspect ratio
                text_regions.append({
                    'bbox': [x, y, x+w, y+h],
                    'confidence': 0.65,
                    'detector': 'contour'
                })
        
        return text_regions
    
    def _combine_detections(self, detections_list):
        """Combine and deduplicate detections"""
        combined = []
        
        for detections in detections_list:
            for det in detections:
                if isinstance(det, list) and len(det) >= 2:
                    # PaddleOCR format
                    bbox = det[0]
                    text = det[1][0]
                    confidence = det[1][1]
                    
                    # Convert bbox to [x1,y1,x2,y2]
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    
                    combined.append({
                        'bbox': [int(min(x_coords)), int(min(y_coords)), 
                                int(max(x_coords)), int(max(y_coords))],
                        'text': text,
                        'confidence': confidence,
                        'detector': 'paddle'
                    })
                    
                elif isinstance(det, tuple) and len(det) == 3:
                    # EasyOCR format
                    bbox, text, confidence = det
                    combined.append({
                        'bbox': [int(bbox[0][0]), int(bbox[0][1]), 
                                int(bbox[2][0]), int(bbox[2][1])],
                        'text': text,
                        'confidence': confidence,
                        'detector': 'easyocr'
                    })
                    
                elif isinstance(det, dict):
                    # MSER/Contour format
                    combined.append(det)
        
        # Remove overlapping detections
        return self._non_max_suppression(combined)
    
    def _non_max_suppression(self, detections, iou_threshold=0.5):
        """Remove overlapping detections"""
        if not detections:
            return []
            
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            detections = [d for d in detections if self._compute_iou(
                best['bbox'], d['bbox']) < iou_threshold]
        
        return keep
    
    def _compute_iou(self, box1, box2):
        """Compute Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
