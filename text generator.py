# backend/app/models/text_generator.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
import logging
from typing import Dict, Tuple
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline

logger = logging.getLogger(__name__)

class TextGenerator:
    def __init__(self, config):
        self.config = config
        
        # Load font database
        self.fonts = self._load_font_database()
        
        # Load AI text generator (optional)
        self.ai_generator = self._load_ai_generator()
        
    def _load_font_database(self):
        """Load available fonts"""
        fonts = {
            'Arial': 'arial.ttf',
            'Times New Roman': 'times.ttf',
            'Courier New': 'cour.ttf',
            'Impact': 'impact.ttf',
            'Hind Siliguri': 'HindSiliguri-Regular.ttf',  # Bengali font
            'Noto Sans Bengali': 'NotoSansBengali-Regular.ttf',
        }
        return fonts
    
    def _load_ai_generator(self):
        """Load AI model for text generation"""
        try:
            # Use Stable Diffusion for text generation
            pipe = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-I-XL-v1.0",
                torch_dtype=torch.float16
            )
            return pipe
        except:
            logger.warning("AI generator not available, using PIL fallback")
            return None
    
    async def generate_text(self, text: str, style: Dict, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Generate text with exact style matching
        """
        # Try AI generation first
        if self.ai_generator is not None and style.get('ai_generate', False):
            text_image = await self._generate_ai_text(text, style, target_size)
            if text_image is not None:
                return text_image
        
        # Fallback to PIL generation
        return await self._generate_pil_text(text, style, target_size)
    
    async def _generate_pil_text(self, text: str, style: Dict, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Generate text using PIL with advanced features
        """
        w, h = target_size
        
        # Create blank image
        img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Select font
        font_family = style.get('font_family', 'Arial')
        font_path = self.fonts.get(font_family, self.fonts['Arial'])
        
        try:
            # Load font with size
            font_size = style.get('font_size', h)
            font = ImageFont.truetype(font_path, font_size)
        except:
            # Fallback to default
            font = ImageFont.load_default()
        
        # Get text size
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate position to center
        x = (w - text_width) // 2
        y = (h - text_height) // 2
        
        # Get text color
        color = tuple(style.get('color', [255, 255, 255]))
        if len(color) == 3:
            color = (*color, 255)
        
        # Apply text effects
        if style.get('effects', {}).get('has_shadow'):
            # Draw shadow
            shadow_offset = style['effects'].get('shadow_offset', (2, 2))
            shadow_color = (0, 0, 0, 128)
            draw.text((x + shadow_offset[0], y + shadow_offset[1]), 
                     text, font=font, fill=shadow_color)
        
        if style.get('effects', {}).get('has_outline'):
            # Draw outline
            outline_color = style['effects'].get('outline_color', (0, 0, 0, 255))
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        
        # Draw main text
        draw.text((x, y), text, font=font, fill=tuple(color))
        
        # Apply rotation if needed
        angle = style.get('angle', 0)
        if angle != 0:
            img = img.rotate(angle, expand=True, center=(w//2, h//2))
        
        # Convert to numpy array
        result = np.array(img)
        
        # Remove alpha channel if needed
        if result.shape[2] == 4:
            result = result[:, :, :3]
        
        return result
    
    async def _generate_ai_text(self, text: str, style: Dict, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Generate text using AI for photorealistic results
        """
        try:
            # Create prompt
            prompt = f"High quality text '{text}' in {style.get('font_family', 'Arial')} font, {style.get('font_weight', 'normal')} weight, color {style.get('color', 'white')}, photorealistic, sharp, clear"
            
            # Generate
            with torch.no_grad():
                image = self.ai_generator(
                    prompt,
                    negative_prompt="blurry, distorted, low quality",
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    height=target_size[1],
                    width=target_size[0]
                ).images[0]
            
            # Convert to numpy
            result = np.array(image)
            
            # Resize if needed
            if result.shape[:2] != target_size:
                result = cv2.resize(result, target_size)
            
            return result
            
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            return None
