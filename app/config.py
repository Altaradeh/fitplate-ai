"""
Configuration settings for the FitPlate AI application.
"""

"""
Configuration settings for the FitPlate AI application.
Centralize model names and generation defaults to avoid hardcoding.
"""

# OpenAI Model Configuration
VISION_MODEL = "gpt-4o"        # Vision model for image analysis  
CHAT_MODEL = "gpt-4o-mini"     # Faster/lower-cost model for text-only steps

# Speed Optimizations
FAST_MODE = True               # Enable speed optimizations
MAX_FOOD_ITEMS = 8             # Limit items to process for speed
REDUCED_TOKENS = True          # Use smaller token limits for speed

# Generation defaults with speed optimizations
JSON_TEMPERATURE = 0.1
JSON_MAX_TOKENS = 1000 if REDUCED_TOKENS else 1500  # Reduced for speed
JSON_MAX_TOKENS_NUTRITION = 800 if REDUCED_TOKENS else 1200  # Even more reduced for nutrition

TEXT_TEMPERATURE = 0.1
TEXT_MAX_TOKENS = 300 if REDUCED_TOKENS else 400  # Reduced for speed