"""Test script for nutrition analysis functionality."""

import json
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
import os

from services.analyzer import FoodAnalyzerService, AIResponseError
from app.models import FoodItem, NutritionAnalysis, NutritionInfo
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Find the project root directory and load .env
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
logger.info(f"Loading .env from: {env_path}")
load_dotenv(dotenv_path=env_path)
    # Load environment variables
# Load environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Get API key after loading .env
async def test_nutrition_analysis():
    """Test the nutrition analysis functionality."""
    logger.info("Starting nutrition analysis test")
    logger.info("API Key status: %s", "Present" if api_key else "Missing")

    # Initialize the analyzer
    analyzer = FoodAnalyzerService(api_key=api_key)

    # Test data
    logger.info("Creating test food items")
    test_items = [
        FoodItem(
            name="Grilled Chicken",
            type="main_dish",
            quantity="1 piece",
            confidence=0.95
        ),
        FoodItem(
            name="Brown Rice",
            type="side_dish",
            quantity="1 cup",
            confidence=0.9
        ),
        FoodItem(
            name="Steamed Broccoli",
            type="side_dish",
            quantity="1 cup",
            confidence=0.85
        )
    ]

    try:
        logger.info("Starting nutrition analysis test")
        logger.info(f"Test items: {[item.name for item in test_items]}")

        # Test the nutrition analysis
        nutrition_result = await analyzer._analyze_nutrition(test_items)

        if nutrition_result:
            logger.info("Nutrition analysis successful!")
            logger.info("\nCombined Nutrition:")
            logger.info(f"Calories: {nutrition_result.combined.calories}")
            logger.info(f"Protein: {nutrition_result.combined.protein_g}g")
            logger.info(f"Carbs: {nutrition_result.combined.carbs_g}g")
            logger.info(f"Fat: {nutrition_result.combined.fat_g}g")
            
            logger.info("\nIndividual Items:")
            for item in nutrition_result.items:
                logger.info(f"\n{item.name}:")
                logger.info(f"Calories: {item.nutrition.calories}")
                logger.info(f"Protein: {item.nutrition.protein_g}g")
                logger.info(f"Carbs: {item.nutrition.carbs_g}g")
                logger.info(f"Fat: {item.nutrition.fat_g}g")
            
            # Test getting recommendations
            logger.info("\nTesting meal recommendations...")
            recommendations = await analyzer._get_recommendations(nutrition_result)
            
            if recommendations:
                logger.info("Recommendations received successfully!")
                logger.info("\nRecommendations:")
                logger.info(f"Health Score: {recommendations['health_score']}")
                logger.info(f"Meal Type: {recommendations['meal_type']}")
                logger.info("Recommendations:")
                for rec in recommendations['recommendations']:
                    logger.info(f"- {rec}")
                logger.info("\nDietary Considerations:")
                for consideration in recommendations['dietary_considerations']:
                    logger.info(f"- {consideration}")
                
                # Validate recommendation fields
                assert isinstance(recommendations['recommendations'], list), "Recommendations should be a list"
                assert isinstance(recommendations['health_score'], (int, float)), "Health score should be a number"
                assert 0 <= recommendations['health_score'] <= 100, "Health score should be between 0 and 100"
                assert isinstance(recommendations['meal_type'], str), "Meal type should be a string"
                assert recommendations['meal_type'].lower() in {'breakfast', 'lunch', 'dinner', 'snack'}, "Invalid meal type"
                assert isinstance(recommendations['dietary_considerations'], list), "Dietary considerations should be a list"
                assert len(recommendations['recommendations']) > 0, "Should have at least one recommendation"
                assert len(recommendations['dietary_considerations']) > 0, "Should have at least one dietary consideration"
                
                logger.info("Recommendation validation passed!")
            else:
                logger.error("Recommendations returned None")
        else:
            logger.error("Nutrition analysis returned None")

    except AIResponseError as e:
        logger.error(f"AI Response Error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Run the test."""
    asyncio.run(test_nutrition_analysis())

if __name__ == "__main__":
    main()