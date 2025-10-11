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

# Generation defaults
# Lower temperature and bounded max_tokens to reduce latency and improve determinism
JSON_TEMPERATURE = 0.1
JSON_MAX_TOKENS = 1500  # Increased to handle meals with many items (9+ components)

TEXT_TEMPERATURE = 0.1
TEXT_MAX_TOKENS = 400