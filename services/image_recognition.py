"""
Image recognition service using OpenAI's Vision API to identify food dishes from images.
"""
import base64
import io
import logging
import os
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_key")
if not api_key:
    logger.error("No OpenAI API key found in environment variables!")
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")

client = AsyncOpenAI(api_key=api_key)
logger.info("OpenAI client initialized successfully")

# Model configuration
from app.config import VISION_MODEL  # Import vision model from config

class ImageRecognitionResult:
    """Structured result from image recognition including confidence scores."""
    def __init__(self, dish_name: str, confidence: float, alternatives: Optional[List[Dict]] = None):
        self.dish_name = dish_name
        self.confidence = confidence  # 0.0 to 1.0
        self.alternatives = alternatives or []  # List of {name: str, confidence: float}

async def analyze_image(image: Image.Image, retry_with_alternatives: bool = True) -> ImageRecognitionResult:
    """
    Analyze a food image using OpenAI's Vision API and return structured results.
    
    Args:
        image: PIL Image object to analyze
        retry_with_alternatives: If True and confidence is low, make a second call
            requesting alternative possibilities
    
    Returns:
        ImageRecognitionResult with dish name, confidence, and alternatives
    """
    # Convert PIL Image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # First pass: Get primary dish identification
    system_msg = (
        "You are an expert food recognition system. Given an image, identify the "
        "primary dish shown with high accuracy. Respond ONLY with a JSON object containing:\n"
        "- dish_name: The most specific, accurate name for the dish\n"
        "- confidence: Your confidence score from 0.0 to 1.0\n"
        "- cuisine: The likely cuisine origin (optional)\n\n"
        "Example: {\"dish_name\": \"Chicken Tikka Masala\", \"confidence\": 0.92, \"cuisine\": \"Indian\"}"
    )
    
    user_msg = f"Analyze this image and identify the dish shown. base64: {img_str}"
    
    try:
        logger.info("Making API call to OpenAI Vision...")
        response = await client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": user_msg},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
                        }
                    ]
                }
            ],
            max_tokens=150,
            temperature=0.1
        )
        logger.info("Successfully received response from OpenAI")
        
        # Parse the response
        result = parse_vision_response(response)
        logger.info(f"Parsed result: {result.dish_name} (confidence: {result.confidence})")
        
        # If confidence is low and retry is enabled, get alternatives
        if retry_with_alternatives and result.confidence < 0.8:
            logger.info("Low confidence, getting alternative suggestions...")
            alternatives = await get_alternative_dishes(image, result.dish_name)
            result.alternatives = alternatives
            
            # If an alternative has higher confidence, swap it
            if alternatives and alternatives[0]["confidence"] > result.confidence:
                logger.info("Found better alternative, updating result...")
                result.dish_name = alternatives[0]["name"]
                result.confidence = alternatives[0]["confidence"]
                alternatives.pop(0)  # Remove from alternatives
                
        return result
        
    except Exception as e:
        logger.error(f"Error in image analysis: {str(e)}", exc_info=True)
        return ImageRecognitionResult("Unknown Dish", 0.0)

async def get_alternative_dishes(image: Image.Image, primary_guess: str) -> List[Dict]:
    """Make a second API call requesting alternative possibilities."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    system_msg = (
        f"The image was initially identified as {primary_guess}. Please list 2-3 other "
        "possible dishes this could be, with confidence scores. Format as JSON array:\n"
        "[{\"name\": \"dish name\", \"confidence\": 0.85}, ...]\n"
        "Focus on visually similar dishes that could be confused with the primary guess."
    )
    
    try:
        response = await client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What other dishes could this be?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
                        }
                    ]
                }
            ],
            max_tokens=150,
            temperature=0.2
        )
        
        # Extract and parse alternatives from response
        if response.choices:
            text = response.choices[0].message.content or ""
            import re
            import json
            # Find JSON array in response
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if m:
                try:
                    alternatives = json.loads(m.group(0))
                    return [
                        {"name": alt["name"], "confidence": float(alt["confidence"])}
                        for alt in alternatives
                        if isinstance(alt, dict) and "name" in alt and "confidence" in alt
                    ]
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass
                    
        return []  # Return empty list if parsing fails
        
    except Exception as e:
        print(f"Error getting alternatives: {e}")
        return []

def parse_vision_response(response) -> ImageRecognitionResult:
    """Parse the OpenAI Vision API response into structured results."""
    if not response.choices:
        return ImageRecognitionResult("Unknown Dish", 0.0)
        
    text = response.choices[0].message.content or ""
    
    # Try to extract JSON from response
    import re
    import json
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            name = data.get("dish_name", "Unknown Dish")
            confidence = float(data.get("confidence", 0.0))
            # Clamp confidence between 0 and 1
            confidence = max(0.0, min(1.0, confidence))
            return ImageRecognitionResult(name, confidence)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
            
    # Fallback: Basic text parsing
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    name = lines[0] if lines else "Unknown Dish"
    
    # Look for a confidence score
    confidence = 0.0
    for line in lines:
        if "confidence" in line.lower():
            matches = re.findall(r"0\.\d+", line)
            if matches:
                try:
                    confidence = float(matches[0])
                except ValueError:
                    pass
                    
    return ImageRecognitionResult(name, confidence)
