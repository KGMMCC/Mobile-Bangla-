# backend/app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
from PIL import Image
import io
import base64
import torch
import logging
from typing import List, Dict
import uvicorn

from app.models.ocr_model import OCRProcessor
from app.models.inpainting import InpaintingModel
from app.models.style_matcher import StyleMatcher
from app.models.text_generator import TextGenerator
from app.services.image_processor import ImageProcessor
from app.utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Professional Text Editor API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
config = Config()
ocr_processor = OCRProcessor(config)
inpainting_model = InpaintingModel(config)
style_matcher = StyleMatcher(config)
text_generator = TextGenerator(config)
image_processor = ImageProcessor(config)

@app.post("/api/detect-text")
async def detect_text(file: UploadFile = File(...)):
    """
    Detect all text in image with high accuracy
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Advanced text detection
        text_blocks = await ocr_processor.detect_text_advanced(img_rgb)
        
        # Analyze each text block for style
        for block in text_blocks:
            # Extract text region
            x1, y1, x2, y2 = block['bbox']
            text_region = img_rgb[y1:y2, x1:x2]
            
            # Detect font style
            block['style'] = await style_matcher.analyze_text_style(text_region)
            
            # Detect text color (with shadow detection)
            block['color'] = await style_matcher.detect_text_color(text_region)
            
            # Detect text angle/rotation
            block['angle'] = await style_matcher.detect_text_angle(text_region)
            
            # Detect font size
            block['font_size'] = y2 - y1
            
            # Detect if text has effects (shadow, outline)
            block['effects'] = await style_matcher.detect_text_effects(text_region)
        
        return JSONResponse({
            'success': True,
            'text_blocks': text_blocks,
            'image_size': img.shape[:2]
        })
        
    except Exception as e:
        logger.error(f"Error in text detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/replace-text")
async def replace_text(
    image: str = None,  # base64
    file: UploadFile = None,
    text_edit: dict = None
):
    """
    Replace text in image with perfect background reconstruction
    """
    try:
        # Load image
        if image:
            # From base64
            img_data = base64.b64decode(image.split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            # From file
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get edit parameters
        bbox = text_edit['bbox']
        old_text = text_edit['old_text']
        new_text = text_edit['new_text']
        style = text_edit['style']
        
        # STEP 1: Remove old text with advanced inpainting
        img = await inpainting_model.remove_text_perfect(img, bbox)
        
        # STEP 2: Generate new text with matched style
        text_overlay = await text_generator.generate_text(
            text=new_text,
            style=style,
            target_size=(bbox[2]-bbox[0], bbox[3]-bbox[1])
        )
        
        # STEP 3: Blend text perfectly with background
        result = await image_processor.blend_text_perfect(
            img, 
            text_overlay, 
            (bbox[0], bbox[1]),
            style
        )
        
        # STEP 4: Apply final enhancements
        result = await image_processor.enhance_final_image(result)
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', result)
        img_base64 = base64.b64encode(buffer).decode()
        
        return JSONResponse({
            'success': True,
            'edited_image': f'data:image/png;base64,{img_base64}'
        })
        
    except Exception as e:
        logger.error(f"Error in text replacement: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch-edit")
async def batch_edit(file: UploadFile = File(...), edits: List[dict] = None):
    """
    Edit multiple text blocks at once
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Sort edits from bottom to top (to handle overlapping text)
        edits = sorted(edits, key=lambda x: x['bbox'][3], reverse=True)
        
        for edit in edits:
            # Remove old text
            img = await inpainting_model.remove_text_perfect(img, edit['bbox'])
            
            # Generate new text
            text_overlay = await text_generator.generate_text(
                text=edit['new_text'],
                style=edit['style'],
                target_size=(edit['bbox'][2]-edit['bbox'][0], 
                           edit['bbox'][3]-edit['bbox'][1])
            )
            
            # Blend
            img = await image_processor.blend_text_perfect(
                img, 
                text_overlay, 
                (edit['bbox'][0], edit['bbox'][1]),
                edit['style']
            )
        
        # Final enhancement
        img = await image_processor.enhance_final_image(img)
        
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode()
        
        return JSONResponse({
            'success': True,
            'edited_image': f'data:image/png;base64,{img_base64}'
        })
        
    except Exception as e:
        logger.error(f"Error in batch edit: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auto-enhance")
async def auto_enhance(file: UploadFile = File(...)):
    """
    Auto-enhance image quality and text readability
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Auto enhance
        enhanced = await image_processor.auto_enhance(img)
        
        _, buffer = cv2.imencode('.png', enhanced)
        img_base64 = base64.b64encode(buffer).decode()
        
        return JSONResponse({
            'success': True,
            'enhanced_image': f'data:image/png;base64,{img_base64}'
        })
        
    except Exception as e:
        logger.error(f"Error in auto enhance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
