"""Food Analysis Service - Handles all AI and analysis logic."""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
from pydantic import ValidationError

from openai import AsyncOpenAI
from app.models import (
    DishAnalysis, NutritionAnalysis, FoodItem,
    NutritionInfo, FoodItemWithNutrition,
    VisionAnalysisResponse, NutritionAnalysisResponse,
    MealRecommendations
)

# Configure service logger
logger = logging.getLogger(__name__)

class AIResponseError(Exception):
    """Custom exception for AI response handling errors."""
    pass

class FoodAnalyzerService:
    def __init__(self, api_key: str) -> None:
        """Initialize the FoodAnalyzer with API key."""
        from app.config import VISION_MODEL, CHAT_MODEL
        self.api_key = api_key
        self.client = AsyncOpenAI(api_key=api_key)
        self.VISION_MODEL = VISION_MODEL
        self.CHAT_MODEL = CHAT_MODEL

    def _clean_ai_response(self, response: str) -> str:
        """Clean and validate AI response.
        
        Handles markdown-wrapped responses and simple text cleanup.
        Relies on Pydantic for actual JSON validation and parsing.
        """
        if not response:
            raise AIResponseError("Empty response received")
            
        # Simple cleanup of markdown and whitespace
        cleaned = response.strip()
        
        # Remove markdown code blocks if present
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned[3:]
            if cleaned.startswith("json\n"):
                cleaned = cleaned[5:]
            cleaned = cleaned[:-3].strip()
            
        return cleaned

    async def analyze_meal(self, image: Image.Image) -> Dict[str, Any]:
        """Complete meal analysis returning structured data for UI rendering."""
        try:
            # Step 1: Analyze the image
            scene_analysis = await self._analyze_image(image)
            if not scene_analysis or not scene_analysis.items:
                logger.error("No items detected in image analysis")
                return {
                    "status": "error",
                    "error": "Could not detect any food items in the image"
                }
            
            # Step 2: Get nutrition information
            nutrition = await self._analyze_nutrition(scene_analysis.items)
            if not nutrition:
                logger.error("Failed to analyze nutrition")
                return {
                    "status": "error",
                    "error": "Could not analyze nutritional information"
                }
            
            # Step 3: Get AI recommendations
            recommendations = await self._get_recommendations(nutrition)
            if not recommendations:
                logger.warning("Could not generate recommendations")
                recommendations = {
                    "recommendations": ["Unable to generate specific recommendations"],
                    "health_score": 50,
                    "meal_type": "unknown",
                    "dietary_considerations": []
                }
            
            # Step 4: Structure the response
            return {
                "status": "success",
                "meal_info": {
                    "name": scene_analysis.items[0].name if scene_analysis.items else "Unknown Dish",
                    "confidence": scene_analysis.items[0].confidence if scene_analysis.items else 0.0,
                    "serving_size": nutrition.combined.serving_size
                },
                "nutrition_summary": {
                    "calories": {
                        "value": nutrition.combined.calories,
                        "daily_value": int((nutrition.combined.calories/2000)*100)
                    },
                    "macros": {
                        "protein": {
                            "value": nutrition.combined.protein_g,
                            "daily_value": int((nutrition.combined.protein_g/50)*100)
                        },
                        "carbs": {
                            "value": nutrition.combined.carbs_g,
                            "daily_value": int((nutrition.combined.carbs_g/225)*100)
                        },
                        "fat": {
                            "value": nutrition.combined.fat_g,
                            "daily_value": int((nutrition.combined.fat_g/65)*100)
                        }
                    },
                    "additional": {
                        "fiber": {
                            "value": nutrition.combined.fiber_g,
                            "daily_value": int((nutrition.combined.fiber_g/28)*100)
                        },
                        "sugar": {
                            "value": nutrition.combined.sugar_g,
                            "warning": nutrition.combined.sugar_g > 20
                        },
                        "saturated_fat": {
                            "value": nutrition.combined.saturated_fat_g,
                            "daily_value": int((nutrition.combined.saturated_fat_g/20)*100)
                        }
                    },
                    "diet_tags": nutrition.combined.diet_tags,
                    "warnings": nutrition.combined.warnings
                },
                "components": [
                    {
                        "name": item.name,
                        "type": item.type,
                        "serving": item.nutrition.serving_size,
                        "nutrition": {
                            "calories": item.nutrition.calories,
                            "protein": item.nutrition.protein_g,
                            "carbs": item.nutrition.carbs_g,
                            "fat": item.nutrition.fat_g,
                            "fiber": item.nutrition.fiber_g,
                            "sugar": item.nutrition.sugar_g,
                            "saturated_fat": item.nutrition.saturated_fat_g
                        },
                        "diet_tags": item.nutrition.diet_tags
                    }
                    for item in nutrition.items
                ],
                "ai_insights": {
                    "recommendations": recommendations["recommendations"],
                    "health_score": recommendations["health_score"],
                    "meal_type": recommendations["meal_type"],
                    "dietary_considerations": recommendations["dietary_considerations"]
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing meal: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _analyze_image(self, image: Image.Image) -> DishAnalysis:
        """Analyze image to identify food items."""
        try:
            # Convert image to base64
            import base64
            import io
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Get JSON schema from our Pydantic model
            vision_schema = VisionAnalysisResponse.model_json_schema()
            
            # Send to OpenAI Vision API with explicit schema
            prompt = f"""Analyze this food image and list all visible food items and beverages.
            Your response MUST conform to this JSON schema:
            
            {json.dumps(vision_schema, indent=2)}
            
            Key requirements:
            - name: Must be a non-empty string
            - type: Must be one of: main_dish, side_dish, beverage, condiment
            - quantity: Should be a human-readable portion size
            - confidence: Must be a number between 0.0 and 1.0
            
            Return ONLY valid JSON matching this schema."""

            # Prepare API call with strict JSON response format
            response = await self.client.chat.completions.create(
                model=self.VISION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise food analysis system. "
                            "Analyze the image and return ONLY a valid JSON object matching the provided schema. "
                            "Do not include any markdown formatting, explanations, or additional text."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_str}"
                                }
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.3,  # Lower temperature for more consistent formatting
                max_tokens=1000
            )

            # Parse and validate response using Pydantic
            try:
                if not response.choices or not response.choices[0].message:
                    raise AIResponseError("No response content received from API")
                
                raw_response = response.choices[0].message.content
                
                # Clean and validate the response
                raw_response = self._clean_ai_response(raw_response)
                logger.debug(f"Cleaned response for parsing: {raw_response}")
                
                # Validate through Pydantic
                try:
                    vision_response = VisionAnalysisResponse.model_validate_json(
                        raw_response,
                        strict=True
                    )
                except ValidationError as ve:
                    logger.error("Schema validation failed:")
                    for error in ve.errors():
                        logger.error(f"Field: {error['loc']}, Error: {error['msg']}")
                    raise AIResponseError("Response failed schema validation") from ve
                except json.JSONDecodeError as je:
                    logger.error(f"JSON parsing failed: {str(je)}")
                    logger.debug(f"Attempted to parse: {raw_response}")
                    raise AIResponseError("Invalid JSON in response") from je
                
                # Log success with item count
                item_count = len(vision_response.items)
                logger.info(f"Successfully validated response with {item_count} food items")
                
                return DishAnalysis(items=vision_response.items)
                
            except AIResponseError as are:
                logger.error(f"AI response error: {str(are)}")
                return DishAnalysis(items=[
                    FoodItem(
                        name="Unidentified Food",
                        type="main_dish",
                        confidence=0.0
                    )
                ])
            except Exception as e:
                logger.error(f"Unexpected error in vision analysis: {str(e)}")
                return DishAnalysis(items=[
                    FoodItem(
                        name="Unidentified Food",
                        type="main_dish",
                        confidence=0.0
                    )
                ])
                
        except Exception as e:
            logger.error(f"Error in image analysis: {str(e)}")
            return DishAnalysis(items=[
                FoodItem(
                    name="Unidentified Food",
                    type="main_dish",
                    confidence=0.0
                )
            ])

    async def _get_recommendations(self, nutrition: NutritionAnalysis) -> MealRecommendations:
        """Get meal recommendations based on nutritional analysis."""
        try:
            logger.info("Getting meal recommendations")
            
            # Prepare the nutrition data for the prompt
            items_summary = "\n".join([
                f"- {item.name}: {item.nutrition.calories}cal, "
                f"{item.nutrition.protein_g}g protein, "
                f"{item.nutrition.carbs_g}g carbs, "
                f"{item.nutrition.fat_g}g fat"
                for item in nutrition.items
            ])
            
            combined = nutrition.combined
            prompt = (
                "Analyze this meal and provide recommendations in this exact JSON format:\n"
                "{\n"
                "  'meal_rating': 8.5,  // Overall rating out of 10\n"
                "  'health_score': 85,  // Health score out of 100\n"
                "  'suggestions': ['Add more vegetables', 'Consider whole grains'],  // List of improvements\n"
                "  'improvements': ['Increase fiber intake', 'Reduce saturated fat'],  // Specific improvements\n"
                "  'positive_aspects': ['Good protein content', 'Balanced macros']  // Positive points\n"
                "}\n\n"
                f"Meal Analysis:\n{items_summary}\n\n"
                f"Combined Nutrition:\n"
                f"- Total Calories: {combined.calories}\n"
                f"- Protein: {combined.protein_g}g\n"
                f"- Carbs: {combined.carbs_g}g\n"
                f"- Fat: {combined.fat_g}g\n"
                f"- Fiber: {combined.fiber_g}g\n"
                f"- Sugar: {combined.sugar_g}g\n"
                f"- Saturated Fat: {combined.saturated_fat_g}g\n"
            )

            # Get recommendations from GPT-4
            response = await self.client.chat.completions.create(
                model=self.CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional nutritionist providing meal recommendations. "
                            "Respond with valid JSON only, using the exact format shown. "
                            "Give specific, actionable advice based on the nutritional analysis."
                        )
                    },
                    {"role": "user", "content": prompt}
                ]
            )

            # Parse and validate the response
            raw_response = self._clean_ai_response(response.choices[0].message.content)
            logger.debug("Raw recommendation response:")
            logger.debug(raw_response)

            try:
                recommendations_data = json.loads(raw_response)
                logger.debug("Parsed recommendations:")
                logger.debug(json.dumps(recommendations_data, indent=2))
                
                # Validate with Pydantic
                recommendations = MealRecommendations.model_validate(recommendations_data)
                logger.info("Recommendations validation successful")
                return recommendations
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse recommendations JSON: {e}")
                raise AIResponseError("Invalid recommendations format received")
            except ValidationError as ve:
                logger.error(f"Recommendations validation error: {ve}")
                raise AIResponseError(f"Invalid recommendations data: {ve}")
                
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            raise AIResponseError(f"Failed to process recommendations: {str(e)}")

    async def _analyze_nutrition(self, items: List[FoodItem]) -> NutritionAnalysis:
        """Analyze nutrition for identified items."""
        if not items:
            logger.error("No items to analyze")
            unidentified = FoodItemWithNutrition(
                name="Unidentified Food",
                type="main_dish",
                nutrition=NutritionInfo(
                    calories=0,
                    serving_size="N/A",
                    protein_g=0,
                    carbs_g=0,
                    fiber_g=0,
                    sugar_g=0,
                    fat_g=0,
                    saturated_fat_g=0,
                    category="unknown",
                    diet_tags=[],
                    warnings=["Could not identify food items"]
                )
            )
            return NutritionAnalysis(
                items=[unidentified],
                combined=NutritionInfo(
                    calories=0,
                    serving_size="N/A",
                    protein_g=0,
                    carbs_g=0,
                    fiber_g=0,
                    sugar_g=0,
                    fat_g=0,
                    saturated_fat_g=0,
                    category="unknown",
                    diet_tags=[],
                    warnings=["Could not identify food items"]
                )
            )

        try:
            # Log items being processed
            item_names = [f"{item.name} ({item.type})" for item in items]
            logger.info(f"Input items for nutrition analysis: {[item.name for item in items]}")
            
            for item in items:
                logger.info(f"Processing item: {item.name}")

            # Convert items to format for API
            api_items = [
                {
                    "name": item.name,
                    "type": item.type,
                    "quantity": item.quantity
                }
                for item in items
            ]

            logger.info(f"Estimating nutrition for items: {item_names}")
            
            # Prepare prompt for GPT-4
            prompt = (
                "Analyze these food items and provide nutritional information matching this EXACT format:\n\n"
                f"{json.dumps(api_items, indent=2)}\n\n"
                "Required response format:\n"
                "{\n"
                "  'combined': {\n"
                "    'calories': 800,\n"
                "    'serving_size': '1 meal',\n"
                "    'protein_g': 30,\n"
                "    'carbs_g': 90,\n"
                "    'fiber_g': 12,\n"
                "    'sugar_g': 15,\n"
                "    'fat_g': 35,\n"
                "    'saturated_fat_g': 8,\n"
                "    'category': 'meal',\n"
                "    'diet_tags': ['High-Protein'],\n"
                "    'warnings': []\n"
                "  },\n"
                "  'items': [{\n"
                "    'name': 'Item Name',\n"
                "    'type': 'main_dish',  # must be: main_dish, side_dish, beverage, or condiment\n"
                "    'quantity': '1 serving',\n"
                "    'calories': 300,\n"
                "    'serving_size': '1 serving',\n"
                "    'protein_g': 25,\n"
                "    'carbs_g': 30,\n"
                "    'fiber_g': 4,\n"
                "    'sugar_g': 5,\n"
                "    'fat_g': 12,\n"
                "    'saturated_fat_g': 3,\n"
                "    'category': 'protein',\n"
                "    'diet_tags': ['High-Protein'],\n"
                "    'warnings': []\n"
                "  }]\n"
                "}\n\n"
                "CRITICAL REQUIREMENTS:\n"
                "1. Use EXACTLY these field names (all lowercase)\n"
                "2. type field MUST be one of: main_dish, side_dish, beverage, condiment\n"
                "3. All fields shown are required\n"
                "4. All numeric fields must be numbers (not strings)\n"
                "5. Arrays must be lists even if empty (use [] not null)\n\n"
                "Provide accurate nutritional information using these exact field names and types."
            )
            example_response = {
                "combined": {
                    "calories": 800,
                    "serving_size": "1 meal",
                    "protein_g": 30,
                    "carbs_g": 90,
                    "fiber_g": 12,
                    "sugar_g": 15,
                    "fat_g": 35,
                    "saturated_fat_g": 8,
                    "category": "meal",
                    "diet_tags": ["High-Protein", "Mediterranean"],
                    "warnings": []
                },
                "items": [
                    {
                        "name": "Grilled Chicken",
                        "type": "main_dish",
                        "quantity": "1 piece",
                        "calories": 300,
                        "serving_size": "1 piece",
                        "protein_g": 25,
                        "carbs_g": 0,
                        "fiber_g": 0,
                        "sugar_g": 0,
                        "fat_g": 12,
                        "saturated_fat_g": 3,
                        "category": "protein",
                        "diet_tags": ["High-Protein", "Low-Carb"],
                        "warnings": []
                    }
                ]
            }

            prompt = (
                "Analyze the following food items and provide detailed nutritional information:\n\n"
                f"{json.dumps(api_items, indent=2)}\n\n"
                "Your response MUST be a JSON object with EXACTLY these two top-level keys:\n"
                "1. 'Combined': An object containing combined meal nutrition\n"
                "2. 'Items': An array of objects with individual item nutrition\n\n"
                "The Combined object MUST include:\n"
                "- Calories (integer)\n"
                "- Serving_Size (string, in common measurements)\n"
                "- Protein (number, grams)\n"
                "- Carbs (number, grams)\n"
                "- Fiber (number, grams)\n"
                "- Sugar (number, grams)\n"
                "- Fat (number, grams)\n"
                "- Sat_Fat (number, grams)\n"
                "- Category (string)\n"
                "- Diet_Tags (array of strings)\n"
                "- Warnings (array of strings)\n\n"
                "Each item in the Items array MUST include ALL of the above fields PLUS:\n"
                "- name (string, matching input, required)\n"
                "- type (string, MUST be one of: main_dish, side_dish, beverage, condiment)\n"
                "- quantity (string, portion size)\n\n"
                "Example response structure:\n"
                f"{json.dumps(example_response, indent=2)}\n\n"
                "IMPORTANT:\n"
                "1. All fields are required\n"
                "2. Type field must be exactly one of: main_dish, side_dish, beverage, condiment\n"
                "3. Use consistent units (grams for weight-based measurements)\n"
                "4. Follow the example structure exactly"
            )

            # Get nutrition estimation from GPT-4
            system_message = (
                "You are a professional nutritionist providing accurate nutritional analysis. "
                "CRITICAL REQUIREMENTS:\n"
                "1. Respond with valid JSON only, no other text\n"
                "2. Use exactly the field names shown in the example\n"
                "3. All field names must be lowercase\n"
                "4. Follow the data types exactly (numbers as numbers, strings as strings)\n"
                "5. Include all required fields\n"
                "6. The 'type' field must be one of: main_dish, side_dish, beverage, condiment"
            )
            
            response = await self.client.chat.completions.create(
                model=self.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
            )

            # Get and clean the response
            raw_response = self._clean_ai_response(response.choices[0].message.content)
            logger.debug("Raw GPT response content:")
            logger.debug(response.choices[0].message.content)
            logger.debug("\nCleaned response:")
            logger.debug(raw_response)
            
            def convert_to_numeric(val):
                """Convert string values to appropriate numeric types."""
                if isinstance(val, (int, float)):
                    return val
                if isinstance(val, str):
                    try:
                        if '.' in val:
                            return float(val)
                        return int(val)
                    except ValueError:
                        return val
                return val

            def normalize_keys(data):
                """Normalize dictionary keys to match our model's expectations."""
                if not data:
                    return data
                
                if not isinstance(data, (dict, list)):
                    return data
                    
                if isinstance(data, list):
                    return [normalize_keys(item) for item in data]
                
                # Define key mapping with proper field names
                field_mapping = {
                    'calories': 'calories',
                    'serving_size': 'serving_size',
                    'protein': 'protein_g',
                    'carbs': 'carbs_g',
                    'fiber': 'fiber_g',
                    'sugar': 'sugar_g',
                    'fat': 'fat_g',
                    'sat_fat': 'saturated_fat_g',
                    'saturated_fat': 'saturated_fat_g',
                    'category': 'category',
                    'diet_tags': 'diet_tags',
                    'warnings': 'warnings',
                    'name': 'name',
                    'type': 'type',
                    'quantity': 'quantity',
                    'combined': 'combined',
                    'items': 'items'
                }

                result = {}
                for k, v in data.items():
                    k_lower = k.lower()
                    # Get the model field name from mapping, keeping original if not found
                    new_key = field_mapping.get(k_lower, k)
                    result[new_key] = normalize_keys(v)
                    logger.debug(f"Key normalization: {k} -> {new_key}")
                    
                # Convert numeric values if possible
                if 'calories' in result:
                    result['calories'] = convert_to_numeric(result['calories'])
                for key in ['protein_g', 'carbs_g', 'fiber_g', 'sugar_g', 'fat_g', 'saturated_fat_g']:
                    if key in result:
                        result[key] = convert_to_numeric(result[key])
                        
                return result

            try:
                # Parse the JSON response
                try:
                    parsed_data = json.loads(raw_response)
                    logger.info("\nParsed JSON structure:")
                    logger.info(json.dumps(parsed_data, indent=2))
                    
                    # Check if required top-level keys exist (case-insensitive)
                    available_keys = {k.lower(): k for k in parsed_data.keys()}
                    if 'combined' not in available_keys or 'items' not in available_keys:
                        logger.error("Missing required top-level keys in parsed data")
                        logger.error(f"Available keys: {list(parsed_data.keys())}")
                        raise AIResponseError("Response missing required fields: 'combined' and/or 'items'")
                        
                except json.JSONDecodeError as je:
                    logger.error(f"JSON parsing error at position {je.pos}: {je.msg}")
                    logger.error(f"Error context: {raw_response[max(0, je.pos-50):je.pos+50]}")
                    raise AIResponseError(f"Failed to parse response as JSON: {str(je)}")
                
                # Normalize the data structure
                normalized_data = normalize_keys(parsed_data)
                logger.info("\nNormalized data structure:")
                logger.info(json.dumps(normalized_data, indent=2))
                logger.info(f"Normalized keys: {list(normalized_data.keys())}")

                # Validate and prepare the data structure
                if 'combined' not in normalized_data or 'items' not in normalized_data:
                    logger.error("Missing required keys after normalization")
                    logger.error(f"Available keys: {list(normalized_data.keys())}")
                    raise AIResponseError("Response missing required fields: 'combined' and/or 'items'")

                # Ensure each item in items has all required fields
                required_item_fields = {
                    'name', 'type', 'calories', 'serving_size', 'protein_g', 'carbs_g',
                    'fiber_g', 'sugar_g', 'fat_g', 'saturated_fat_g', 'category'
                }

                for idx, item in enumerate(normalized_data['items']):
                    missing_fields = required_item_fields - set(item.keys())
                    if missing_fields:
                        logger.error(f"Item {idx} is missing required fields: {missing_fields}")
                        logger.error(f"Item content: {json.dumps(item, indent=2)}")
                        raise AIResponseError(f"Item {idx} missing required fields: {missing_fields}")

                # Validate with Pydantic
                try:
                    # Debug the structure before validation
                    logger.info("Data structure before validation:")
                    logger.info(json.dumps(normalized_data, indent=2))
                    
                    # Log validation details
                    if "combined" in normalized_data:
                        combined_data = normalized_data["combined"]
                        logger.info("\nCombined data types:")
                        for k, v in combined_data.items():
                            logger.info(f"{k}: {type(v)} = {v}")
                    
                    if "items" in normalized_data:
                        for i, item in enumerate(normalized_data["items"]):
                            logger.info(f"\nItem {i + 1} data types:")
                            for k, v in item.items():
                                logger.info(f"{k}: {type(v)} = {v}")
                    
                    # Try to validate with Pydantic
                    logger.info("\nAttempting Pydantic validation...")
                    nutrition_response = NutritionAnalysisResponse.model_validate(normalized_data)
                    logger.info("Validation successful!")
                    logger.info(f"Combined nutrition: {nutrition_response.combined}")
                    logger.info(f"Number of items: {len(nutrition_response.items)}")
                    
                    # Log successful validation details
                    logger.debug("Validated response structure:")
                    logger.debug(f"Combined nutrition present: {nutrition_response.combined is not None}")
                    logger.debug(f"Number of items: {len(nutrition_response.items)}")
                    logger.debug(f"First item sample: {nutrition_response.items[0].model_dump() if nutrition_response.items else 'No items'}")
                    
                except ValidationError as ve:
                    logger.error("\n=== VALIDATION ERRORS ===")
                    for error in ve.errors():
                        path = ' -> '.join(str(x) for x in error['loc'])
                        logger.error(f"\nError in field: {path}")
                        logger.error(f"Error type: {error.get('type', 'Unknown')}")
                        logger.error(f"Message: {error['msg']}")
                        
                        # Get the actual value that caused the error
                        current = normalized_data
                        try:
                            for part in error['loc']:
                                current = current[part]
                            logger.error(f"Current value: {current} (type: {type(current)})")
                        except (KeyError, IndexError):
                            logger.error("Could not access current value")
                        
                        # Get the context of where the error occurred
                        if len(error['loc']) > 1:
                            parent_key = error['loc'][-2]
                            logger.error(f"Parent field: {parent_key}")
                            try:
                                parent = normalized_data
                                for part in error['loc'][:-1]:
                                    parent = parent[part]
                                logger.error(f"Parent context: {json.dumps(parent, indent=2)}")
                            except (KeyError, IndexError):
                                logger.error("Could not access parent context")
                    
                    raise AIResponseError("Response failed validation. Check logs for details.") from ve
                
                # Convert API response to internal model format
                analysis = nutrition_response.to_nutrition_analysis()
                
                # Validate the conversion succeeded and contains expected data
                if not analysis:
                    raise AIResponseError("Failed to convert nutrition analysis to internal format")
                if not analysis.combined:
                    raise AIResponseError("Missing combined nutrition information")
                if not analysis.items:
                    raise AIResponseError("Missing individual item nutrition information")
                
                return analysis
                
            except ValidationError as ve:
                logger.error("Nutrition analysis validation failed:")
                for error in ve.errors():
                    logger.error(f"Field: {error['loc']}, Error: {error['msg']}")
                raise AIResponseError("Nutrition analysis failed schema validation") from ve
            except Exception as e:
                logger.error(f"Error processing nutrition response: {str(e)}")
                raise AIResponseError(f"Failed to process nutrition analysis: {str(e)}")

        except Exception as e:
            logger.error(f"Error in nutrition analysis: {str(e)}")
            return None

    async def _get_recommendations(self, nutrition: Optional[NutritionAnalysis]) -> Dict[str, Any]:
        """Get AI-powered recommendations and insights.
        
        Returns:
            Dict[str, Any]: A dictionary containing:
                - recommendations (List[str]): List of specific recommendations
                - health_score (float): Score from 0-100
                - meal_type (str): One of 'breakfast', 'lunch', 'dinner', 'snack'
                - dietary_considerations (List[str]): List of dietary notes
        """
        """Get AI-powered recommendations and insights."""
        try:
            if not nutrition or not nutrition.combined:
                return {
                    "recommendations": ["Unable to generate recommendations without nutrition data"],
                    "health_score": 0,
                    "meal_type": "unknown",
                    "dietary_considerations": ["Nutrition analysis not available"]
                }
                
            # Create a nutrition summary for the AI
            nutrition_summary = {
                "calories": nutrition.combined.calories,
                "protein": nutrition.combined.protein_g,
                "carbs": nutrition.combined.carbs_g,
                "fat": nutrition.combined.fat_g,
                "fiber": nutrition.combined.fiber_g,
                "sugar": nutrition.combined.sugar_g,
                "diet_tags": nutrition.combined.diet_tags
            }

            # Define required fields and their types
            required_fields = {
                "recommendations": {"type": "array of strings", "description": "List of specific, actionable recommendations"},
                "health_score": {"type": "number", "range": "0-100", "description": "Overall nutritional quality score"},
                "meal_type": {"type": "string", "values": ["breakfast", "lunch", "dinner", "snack"], "description": "Suggested meal category"},
                "dietary_considerations": {"type": "array of strings", "description": "Important nutritional aspects to consider"}
            }

            # Create example response format
            example_format = {
                "recommendations": [
                    "Increase protein intake to support muscle maintenance",
                    "Add more fiber-rich vegetables for better satiety"
                ],
                "health_score": 85,
                "meal_type": "lunch",
                "dietary_considerations": [
                    "Moderate in calories",
                    "Good protein content"
                ]
            }

            # Generate field requirements from the required_fields dictionary
            field_requirements = "\n".join([
                f"   - {field}: {specs['type']}"
                f"{f' (valid values: {", ".join(specs["values"])})' if "values" in specs else ''}"
                f"{f' ({specs["range"]})' if "range" in specs else ''}"
                f" - {specs['description']}"
                for field, specs in required_fields.items()
            ])

            # Ask AI for comprehensive analysis
            response = await self.client.chat.completions.create(
                model=self.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are a professional nutritionist providing meal analysis. "
                        "You MUST provide recommendations in EXACTLY this JSON format:\n"
                        f"{json.dumps(example_format, indent=2)}\n\n"
                        "CRITICAL REQUIREMENTS:\n"
                        "1. Include ALL these required fields with exact names and types:\n"
                        f"{field_requirements}\n"
                        "2. Ensure recommendations are specific and actionable\n"
                        "3. Health score must reflect overall nutritional quality\n"
                        "4. Meal type should be suggested based on composition\n"
                        "5. Include at least 2 dietary considerations\n\n"
                        "Respond with valid JSON only, no other text."
                    )},
                    {"role": "user", "content": (
                        "Analyze this meal's nutrition profile and provide recommendations:\n"
                        f"{json.dumps(nutrition_summary, indent=2)}"
                    )}
                ]
            )

            raw_response = self._clean_ai_response(response.choices[0].message.content)
            try:
                recommendations = json.loads(raw_response)
                logger.debug("Raw recommendations response:")
                logger.debug(json.dumps(recommendations, indent=2))
                
                # Use the same required_fields dictionary for validation
                missing_fields = [field for field in required_fields if field not in recommendations]
                if missing_fields:
                    logger.error(f"Missing required fields in recommendations: {missing_fields}")
                    logger.error(f"Available fields: {list(recommendations.keys())}")
                    raise AIResponseError(f"Missing required fields in recommendations response: {missing_fields}")
                
                # Validate types and content based on required_fields specifications
                validation_errors = []
                
                for field, specs in required_fields.items():
                    value = recommendations[field]
                    
                    # Type validation
                    if specs["type"] == "array of strings":
                        if not isinstance(value, list):
                            validation_errors.append(f"{field} must be a list")
                        elif not value:
                            validation_errors.append(f"{field} list cannot be empty")
                        elif not all(isinstance(item, str) for item in value):
                            validation_errors.append(f"all items in {field} must be strings")
                            
                    elif specs["type"] == "number":
                        if not isinstance(value, (int, float)):
                            validation_errors.append(f"{field} must be a number")
                        elif "range" in specs:
                            min_val, max_val = map(int, specs["range"].split("-"))
                            if not min_val <= value <= max_val:
                                validation_errors.append(f"{field} must be between {min_val} and {max_val}")
                                
                    elif specs["type"] == "string":
                        if not isinstance(value, str):
                            validation_errors.append(f"{field} must be a string")
                        elif "values" in specs and value.lower() not in specs["values"]:
                            validation_errors.append(f"{field} must be one of: {', '.join(specs['values'])}")
                
                if validation_errors:
                    logger.error("Validation errors in recommendations:")
                    for error in validation_errors:
                        logger.error(f"- {error}")
                    raise AIResponseError(f"Invalid recommendations format: {'; '.join(validation_errors)}")
                
                # Log successful validation
                logger.info("Recommendations validation successful")
                logger.info(f"Health Score: {recommendations['health_score']}")
                logger.info(f"Meal Type: {recommendations['meal_type']}")
                logger.info(f"Number of recommendations: {len(recommendations['recommendations'])}")
                logger.info(f"Number of dietary considerations: {len(recommendations['dietary_considerations'])}")
                
                return recommendations
                
            except json.JSONDecodeError as je:
                logger.error(f"Failed to parse recommendations response: {str(je)}")
                raise AIResponseError("Invalid recommendations format") from je
            except Exception as e:
                logger.error(f"Error processing recommendations: {str(e)}")
                raise AIResponseError(f"Failed to process recommendations: {str(e)}")

        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return {
                "recommendations": ["Unable to generate recommendations"],
                "health_score": 70,
                "meal_type": "mixed",
                "dietary_considerations": ["Analysis unavailable"]
            }

    async def suggest_improvements(self, meal_data: Dict[str, Any]) -> List[str]:
        """Generate AI-powered meal improvement suggestions based on nutritional analysis."""
        try:
            if not meal_data or "nutrition_summary" not in meal_data:
                return ["Unable to analyze meal: No nutrition data available"]
                
            nutrition = meal_data["nutrition_summary"]
            components = meal_data.get("components", [])
            
            system_msg = (
                "You are an expert nutritionist providing actionable meal improvement suggestions. "
                "Analyze the nutritional data and suggest specific, practical improvements. Consider:\n"
                "1. Macro balance and daily value percentages\n"
                "2. Overall meal composition and component interactions\n"
                "3. Health goals and dietary patterns\n"
                "4. Practical, actionable changes\n\n"
                "Format each suggestion as a clear, concise bullet point starting with an emoji."
            )

            # Prepare context for the AI
            meal_context = {
                "nutrition": {
                    "calories": f"{nutrition['calories']['value']} kcal ({nutrition['calories']['daily_value']}% DV)",
                    "macros": {
                        "protein": f"{nutrition['macros']['protein']['value']}g ({nutrition['macros']['protein']['daily_value']}% DV)",
                        "carbs": f"{nutrition['macros']['carbs']['value']}g ({nutrition['macros']['carbs']['daily_value']}% DV)",
                        "fat": f"{nutrition['macros']['fat']['value']}g ({nutrition['macros']['fat']['daily_value']}% DV)"
                    },
                    "diet_tags": nutrition.get("diet_tags", []),
                },
                "meal_components": [
                    {"name": comp["name"], "type": comp["type"]}
                    for comp in components
                ]
            }

            response = await self.client.chat.completions.create(
                model=self.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"Analyze this meal and suggest improvements:\n{json.dumps(meal_context, indent=2)}"}
                ],
                temperature=0.7,  # Allow some creativity in suggestions
                max_tokens=300
            )

            # Extract suggestions from response
            if response.choices and response.choices[0].message:
                suggestions = [
                    line.strip()
                    for line in response.choices[0].message.content.split("\n")
                    if line.strip() and line.strip().startswith("�")
                ]
                return suggestions or ["✨ This meal has a good nutritional balance!"]
            
            return ["Unable to generate suggestions."]

        except Exception as e:
            logger.error(f"Error generating suggestions: {str(e)}")
            return ["Unable to analyze meal for improvements."]

    async def answer_question(self, question: str, meal_data: Dict[str, Any]):
        """Answer user questions about the meal with structured responses."""
        try:
            # Format nutrition data for the AI
            nutrition = meal_data["nutrition_summary"]
            nutrition_text = ", ".join([
                f"{k}: {v['value']}{' kcal' if k == 'calories' else 'g'}"
                for k, v in nutrition.items()
                if isinstance(v, dict) and 'value' in v
            ])
            if "diet_tags" in nutrition:
                nutrition_text += f", Diet Tags: {', '.join(nutrition['diet_tags'])}"

            system_msg = (
                "You are an expert nutritionist and registered dietitian specializing in personalized meal analysis. "
                "Structure your responses in two parts:\n\n"
                "QUICK ANSWER:\n"
                "- Start with '💡 SUMMARY:'\n"
                "- Provide a 1-2 sentence direct answer\n"
                "- Use emojis for key points\n"
                "- Format key numbers/values in **bold**\n\n"
                "DETAILED ANALYSIS:\n"
                "1. Nutritional Impact:\n"
                "- Analyze macro/micronutrient relevance to question\n"
                "- Reference daily values and guidelines\n"
                "2. Scientific Context:\n"
                "- Cite relevant nutrition principles\n"
                "- Explain metabolic or health implications\n"
                "3. Practical Advice:\n"
                "- Offer specific, actionable recommendations\n"
                "- Suggest modifications if needed\n\n"
                "Keep the detailed section clear and evidence-based while maintaining a conversational tone."
            )

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": (
                    f"The meal is {meal_data['meal_info']['name']}. "
                    f"Full nutritional breakdown: {nutrition_text}. "
                    f"Question: {question}"
                )}
            ]

            # Get streaming response from OpenAI
            stream = await self.client.chat.completions.create(
                model=self.CHAT_MODEL,
                messages=messages,
                stream=True,
                max_tokens=500
            )

            return stream

        except Exception as e:
            logger.error(f"Error in answer_question: {str(e)}")
            raise