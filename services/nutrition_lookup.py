"""
Nutrition lookup service that estimates nutritional values for identified dishes.
Uses a combination of predefined data and AI estimation.
"""
import json
from pathlib import Path
from typing import Dict, Optional

import aiohttp
from openai import AsyncOpenAI
from app.config import CHAT_MODEL

# Initialize OpenAI client
client = AsyncOpenAI()

class NutritionResult:
    """Structured nutrition information for a dish."""
    def __init__(
        self,
        calories: int,
        protein: float,
        carbs: float,
        fat: float,
        source: str = "estimate"
    ):
        self.calories = max(0, int(calories))
        self.protein = max(0, round(float(protein), 1))
        self.carbs = max(0, round(float(carbs), 1))
        self.fat = max(0, round(float(fat), 1))
        self.source = source
        
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "Calories": self.calories,
            "Protein (g)": self.protein,
            "Carbs (g)": self.carbs,
            "Fat (g)": self.fat
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> "NutritionResult":
        """Create from dictionary data."""
        return cls(
            calories=data.get("Calories", 0),
            protein=data.get("Protein (g)", 0),
            carbs=data.get("Carbs (g)", 0),
            fat=data.get("Fat (g)", 0),
            source=data.get("source", "unknown")
        )

class NutritionLookup:
    """Service for looking up or estimating nutrition information."""
    
    def __init__(self, data_path: Optional[str] = None):
        """Initialize with optional path to nutrition database."""
        self.data_path = data_path or str(Path(__file__).parent.parent / "data" / "meals.json")
        self.nutrition_db = self._load_nutrition_db()
        
    def _load_nutrition_db(self) -> Dict:
        """Load nutrition database from JSON file."""
        try:
            with open(self.data_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}  # Return empty dict if file doesn't exist or is invalid
            
    def _save_nutrition_db(self):
        """Save nutrition database to JSON file."""
        with open(self.data_path, 'w') as f:
            json.dump(self.nutrition_db, f, indent=2)
            
    def _normalize_dish_name(self, name: str) -> str:
        """Normalize dish name for consistent lookup."""
        return name.lower().strip()
        
    async def get_nutrition(self, dish_name: str) -> NutritionResult:
        """
        Get nutrition information for a dish, using cached data if available
        or estimating using AI if not.
        """
        normalized_name = self._normalize_dish_name(dish_name)
        
        # Check cache first
        if normalized_name in self.nutrition_db:
            data = self.nutrition_db[normalized_name]
            return NutritionResult.from_dict(data)
            
        # If not in cache, estimate using AI
        nutrition = await self._estimate_nutrition_ai(dish_name)
        
        # Cache the result
        self.nutrition_db[normalized_name] = {
            **nutrition.to_dict(),
            "source": "ai_estimate"
        }
        self._save_nutrition_db()
        
        return nutrition
        
    async def _estimate_nutrition_ai(self, dish_name: str) -> NutritionResult:
        """Use OpenAI to estimate nutrition values for a dish."""
        try:
            system_msg = (
                "You are a nutrition expert. Given a dish name, estimate its typical "
                "nutritional values for a standard serving size. Return ONLY a JSON object "
                "with these keys: Calories (int), Protein (g), Carbs (g), Fat (g).\n\n"
                "Example: {\"Calories\": 350, \"Protein (g)\": 25, \"Carbs (g)\": 30, \"Fat (g)\": 15}"
            )
            
            user_msg = f"Provide nutrition estimates for: {dish_name}"
            
            response = await client.chat.completions.create(
                model=CHAT_MODEL,  # Using configured chat model for nutrition estimates
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            # Extract and parse JSON from response
            if response.choices:
                text = response.choices[0].message.content or ""
                import re
                import json
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if m:
                    try:
                        data = json.loads(m.group(0))
                        return NutritionResult(
                            calories=data.get("Calories", 0),
                            protein=data.get("Protein (g)", 0),
                            carbs=data.get("Carbs (g)", 0),
                            fat=data.get("Fat (g)", 0),
                            source="ai_estimate"
                        )
                    except (json.JSONDecodeError, KeyError):
                        pass
                        
        except Exception as e:
            print(f"Error estimating nutrition: {e}")
            
        # Return conservative default values if estimation fails
        return NutritionResult(
            calories=400,
            protein=20,
            carbs=40,
            fat=15,
            source="default"
        )
