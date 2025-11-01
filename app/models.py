"""Data models for the FitPlate AI application."""

import json
import logging
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from decimal import Decimal

# Configure logger
logger = logging.getLogger(__name__)

# Base Models
class FoodItemBase(BaseModel):
    """Base model for food items."""
    name: str = Field(..., min_length=1, description="Name of the food item")
    type: Literal["main_dish", "side_dish", "beverage", "condiment", "unknown"] = Field(..., description="Type of food item")

    @field_validator('name')
    def validate_name(cls, v: str) -> str:
        """Ensure name is not empty and properly formatted."""
        v = v.strip()
        if not v:
            return "Unknown Item"
        return v.title()

class FoodItem(FoodItemBase):
    """Model for a food item with optional quantity and confidence."""
    quantity: str = Field(default="1 serving", description="Estimated portion or container size")
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the detection"
    )

# Vision Analysis Models
class VisionFoodItem(FoodItemBase):
    """Model for a food item detected in the vision analysis."""
    quantity: str = Field(default="1 serving", description="Estimated portion or container size")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score of the detection")

class VisionAnalysisResponse(BaseModel):
    """Model for the complete vision analysis API response."""
    dish_name: str = Field(..., description="Human-readable name for the overall dish or meal")
    dish_confidence: float = Field(..., ge=0, le=1, description="Confidence score for the dish name")
    items: List[VisionFoodItem] = Field(..., description="List of detected food items")

    @model_validator(mode='after')
    def validate_items(self) -> 'VisionAnalysisResponse':
        """Validate that items list is not empty and all items have reasonable confidence."""
        if not self.items:
            raise ValueError("Must have at least one food item")
        for item in self.items:
            if item.confidence < 0.5:
                logger.warning(f"Low confidence detection for {item.name}: {item.confidence}")
        if not self.dish_name or not isinstance(self.dish_name, str):
            raise ValueError("dish_name must be a non-empty string")
        if not (0.0 <= self.dish_confidence <= 1.0):
            raise ValueError("dish_confidence must be between 0 and 1")
        return self

# Nutrition API Response Models
class NutritionBase(BaseModel):
    """Base model for nutrition information."""
    calories: Union[int, str] = Field(alias="Calories", description="Calories in kcal")
    serving_size: str = Field(alias="Serving_Size", description="Serving size in common measurements")
    protein_g: Union[float, str] = Field(alias="Protein", description="Protein content in grams")
    carbs_g: Union[float, str] = Field(alias="Carbs", description="Carbohydrate content in grams")
    fiber_g: Union[float, str] = Field(alias="Fiber", description="Fiber content in grams")
    sugar_g: Union[float, str] = Field(alias="Sugar", description="Sugar content in grams")
    fat_g: Union[float, str] = Field(alias="Fat", description="Total fat content in grams")
    saturated_fat_g: Union[float, str] = Field(alias="Sat_Fat", description="Saturated fat content in grams")
    category: str = Field(alias="Category", min_length=1, description="Food category")
    diet_tags: Union[List[str], str] = Field(alias="Diet_Tags", default_factory=list, description="Dietary tags")
    warnings: Union[List[str], str] = Field(alias="Warnings", default_factory=list, description="Nutritional warnings")

    class Config:
        """Pydantic model configuration."""
        populate_by_name = True
        validate_by_name = True  # Updated for Pydantic V2

class NutritionAPIResponse(NutritionBase):
    """Model for nutrition information from API with flexible typing."""

    def to_nutrition_info(self) -> 'NutritionInfo':
        """Convert API response to internal NutritionInfo model."""
        def safe_convert_numeric(value, default=0) -> float:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    # Remove any commas and try to convert
                    cleaned = value.replace(',', '').replace('Not identifiable', '0')
                    return float(cleaned)
                except (ValueError, TypeError):
                    pass
            return default

        def safe_convert_list(value, default=None) -> List[str]:
            if default is None:
                default = []
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                if value == 'Not identifiable':
                    return default
                try:
                    if value.startswith('[') and value.endswith(']'):
                        parsed = json.loads(value)
                        if isinstance(parsed, list):
                            return parsed
                    elif ',' in value:
                        return [item.strip() for item in value.split(',')]
                except (json.JSONDecodeError, AttributeError):
                    pass
            return default

        return NutritionInfo(
            calories=int(safe_convert_numeric(self.calories)),
            serving_size=str(self.serving_size) if self.serving_size != 'Not identifiable' else 'N/A',
            protein_g=safe_convert_numeric(self.protein_g),
            carbs_g=safe_convert_numeric(self.carbs_g),
            fiber_g=safe_convert_numeric(self.fiber_g),
            sugar_g=safe_convert_numeric(self.sugar_g),
            fat_g=safe_convert_numeric(self.fat_g),
            saturated_fat_g=safe_convert_numeric(self.saturated_fat_g),
            category=str(self.category) if self.category != 'Not identifiable' else 'unknown',
            diet_tags=safe_convert_list(self.diet_tags),
            warnings=safe_convert_list(self.warnings, ['No specific warnings'])
        )

class NutritionResponseItem(NutritionAPIResponse, FoodItemBase):
    """Model for individual food item nutrition in API responses."""
    quantity: str = Field(default="1 serving", description="Estimated portion or container size")

class NutritionAnalysisResponse(BaseModel):
    """Model for the complete nutrition analysis API response."""
    combined: NutritionAPIResponse = Field(..., description="Combined nutritional information")
    items: List[NutritionResponseItem] = Field(..., description="Individual item nutritional information")

    @model_validator(mode='after')
    def validate_response(self) -> 'NutritionAnalysisResponse':
        """Validate the response structure and data."""
        if not self.items:
            logger.warning("Nutrition analysis response contains no items")
            return self

        def safe_number(value) -> Optional[float]:
            """Safely convert a value to a number."""
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str) and value != 'Not identifiable':
                try:
                    return float(value)
                except ValueError:
                    pass
            return None

        # Validate calories if all values are numeric
        combined_cals = safe_number(self.combined.calories)
        item_cals = [safe_number(item.calories) for item in self.items]
        
        if combined_cals and all(c is not None for c in item_cals):
            total_cals = sum(c for c in item_cals if c is not None)
            if abs(combined_cals - total_cals) > total_cals * 0.2:  # 20% tolerance
                logger.warning(
                    f"Combined calories ({combined_cals}) differ significantly from "
                    f"sum of items ({total_cals})"
                )
        return self

    def to_nutrition_analysis(self) -> 'NutritionAnalysis':
        """Convert API response to internal model."""
        return NutritionAnalysis(
            items=[
                FoodItemWithNutrition(
                    name=item.name,
                    type=item.type,
                    nutrition=item.to_nutrition_info()
                )
                for item in self.items
            ],
            combined=self.combined.to_nutrition_info()
        )

# Internal Models
class NutritionInfo(BaseModel):
    """Internal model for nutritional information with strict typing."""
    calories: int = Field(..., ge=0, lt=5000, description="Calories in kcal")
    serving_size: str = Field(..., min_length=1, description="Serving size in common measurements")
    protein_g: float = Field(..., ge=0, lt=300, description="Protein content in grams")
    carbs_g: float = Field(..., ge=0, lt=500, description="Carbohydrate content in grams")
    fiber_g: float = Field(..., ge=0, lt=100, description="Fiber content in grams")
    sugar_g: float = Field(..., ge=0, lt=200, description="Sugar content in grams")
    fat_g: float = Field(..., ge=0, lt=200, description="Total fat content in grams")
    saturated_fat_g: float = Field(..., ge=0, lt=100, description="Saturated fat content in grams")
    category: str = Field(..., min_length=1, description="Food category")
    diet_tags: List[str] = Field(default_factory=list, description="Dietary tags")
    warnings: List[str] = Field(default_factory=list, description="Nutritional warnings")

    @model_validator(mode='after')
    def validate_measurements(self) -> 'NutritionInfo':
        """Validate nutritional measurements are reasonable."""
        if self.calories < 0 or self.calories > 5000:
            raise ValueError("Calories must be between 0 and 5000")
        if self.protein_g > 300 or self.carbs_g > 500 or self.fat_g > 200:
            raise ValueError("Macronutrient values are unreasonably high")
        return self

class FoodItemWithNutrition(FoodItemBase):
    """Model for a food item with its nutritional information."""
    nutrition: NutritionInfo

class DishAnalysis(BaseModel):
    """Model for the dish analysis response."""
    dish_name: str = Field(..., description="Human-readable name for the overall dish or meal")
    dish_confidence: float = Field(..., ge=0, le=1, description="Confidence score for the dish name")
    items: List[VisionFoodItem] = Field(..., description="List of detected food items")

class MealRecommendations(BaseModel):
    """Model for meal recommendations and analysis."""
    meal_rating: float = Field(..., ge=0, le=10, description="Overall meal rating out of 10")
    health_score: float = Field(..., ge=0, le=100, description="Health score out of 100")
    suggestions: List[str] = Field(..., description="List of suggested improvements")
    improvements: List[str] = Field(..., description="List of specific areas for improvement")
    positive_aspects: List[str] = Field(..., description="List of positive aspects of the meal")

    @field_validator('suggestions')
    def validate_suggestions_not_empty(cls, v: List[str]) -> List[str]:
        """Ensure suggestions list has at least one item."""
        if not v:
            raise ValueError("Suggestions list must have at least one item")
        return v

    @model_validator(mode='after')
    def validate_recommendations(self) -> 'MealRecommendations':
        """Validate recommendation content."""
        if not self.suggestions:
            self.suggestions = ["No specific suggestions"]
        if not self.improvements:
            self.improvements = ["No specific improvements needed"]
        if not self.positive_aspects:
            self.positive_aspects = ["No specific positive aspects noted"]
        return self

class NutritionAnalysis(BaseModel):
    """Model for complete nutrition analysis with typed fields."""
    items: List[FoodItemWithNutrition]
    combined: NutritionInfo

    @model_validator(mode='after')
    def validate_totals(self) -> 'NutritionAnalysis':
        """Validate that combined values make sense with respect to items."""
        if not self.items:
            return self
        
        total_calories = sum(item.nutrition.calories for item in self.items)
        if abs(self.combined.calories - total_calories) > total_calories * 0.1:
            logger.warning(
                f"Combined calories ({self.combined.calories}) differ from "
                f"sum of items ({total_calories}) by more than 10%"
            )
        return self