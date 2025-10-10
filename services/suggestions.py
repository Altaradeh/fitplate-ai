"""
Service for generating personalized dietary suggestions based on meal content and user goals.
"""
from typing import Dict, List, Optional

from pydantic import BaseModel

class UserProfile(BaseModel):
    """User profile with fitness goals and preferences."""
    user_id: str
    goal: str  # "bulking", "cutting", "maintenance"
    target_calories: Optional[int] = None
    target_protein: Optional[int] = None
    allergies: List[str] = []
    preferences: Dict[str, str] = {}

class SuggestionEngine:
    """Engine for generating personalized meal suggestions."""
    
    GOAL_TARGETS = {
        "bulking": {
            "min_protein": 40,  # grams per meal
            "min_calories": 600,
            "protein_warning": "Add more protein for muscle growth.",
            "calorie_warning": "This meal may be too light for bulking.",
        },
        "cutting": {
            "max_calories": 500,
            "min_protein": 30,
            "fat_warning": "Consider a leaner option to reduce calories.",
            "carb_warning": "Try reducing carbs and adding vegetables.",
        },
        "maintenance": {
            "target_calories": 500,
            "min_protein": 25,
            "balance_warning": "Aim for balanced macros: 40% carbs, 30% protein, 30% fat.",
        }
    }

    def generate_suggestions(
        self,
        nutrition: Dict[str, float],
        user_profile: Optional[UserProfile] = None
    ) -> List[str]:
        """
        Generate personalized suggestions based on meal nutrition and user profile.
        
        Args:
            nutrition: Dict with keys "Calories", "Protein (g)", "Carbs (g)", "Fat (g)"
            user_profile: Optional UserProfile for personalized suggestions
            
        Returns:
            List of suggestion strings
        """
        suggestions = []
        
        # Get values from nutrition dict
        calories = nutrition.get("Calories", 0)
        protein = nutrition.get("Protein (g)", 0)
        carbs = nutrition.get("Carbs (g)", 0)
        fat = nutrition.get("Fat (g)", 0)
        
        # Calculate macronutrient ratios
        total_macros = protein * 4 + carbs * 4 + fat * 9  # calories from macros
        if total_macros > 0:
            protein_ratio = (protein * 4 / total_macros) * 100
            carb_ratio = (carbs * 4 / total_macros) * 100
            fat_ratio = (fat * 9 / total_macros) * 100
        else:
            protein_ratio = carb_ratio = fat_ratio = 0
            
        # Basic nutrition suggestions
        if protein < 20:
            suggestions.append(
                "Low protein content. Consider adding: eggs, chicken, fish, "
                "tofu, or legumes."
            )
            
        if fat > 25:
            suggestions.append(
                "High fat content. Try: using less oil, removing visible fat, "
                "or choosing leaner proteins."
            )
            
        if carbs > 75:
            suggestions.append(
                "High carb content. Consider: reducing portion size, adding more "
                "vegetables, or choosing whole grains."
            )
            
        # Goal-specific suggestions if profile provided
        if user_profile:
            goal = user_profile.goal.lower()
            if goal in self.GOAL_TARGETS:
                targets = self.GOAL_TARGETS[goal]
                
                if goal == "bulking":
                    if calories < targets["min_calories"]:
                        suggestions.append(targets["calorie_warning"])
                    if protein < targets["min_protein"]:
                        suggestions.append(targets["protein_warning"])
                        
                elif goal == "cutting":
                    if calories > targets["max_calories"]:
                        suggestions.append(targets["fat_warning"])
                    if carb_ratio > 50:  # If more than 50% calories from carbs
                        suggestions.append(targets["carb_warning"])
                        
                else:  # maintenance
                    if abs(calories - targets["target_calories"]) > 100:
                        suggestions.append(
                            f"Aim for closer to {targets['target_calories']} calories per meal."
                        )
                    if protein < targets["min_protein"]:
                        suggestions.append("Try to include more lean protein sources.")
                        
        # If no issues found, give positive feedback
        if not suggestions:
            if user_profile and user_profile.goal:
                suggestions.append(f"Great meal choice for your {user_profile.goal} goal! 💪")
            else:
                suggestions.append("Well-balanced meal! 👍")
                
        return suggestions

    async def get_meal_analysis(
        self,
        dish_name: str,
        nutrition: Dict[str, float],
        user_profile: Optional[UserProfile] = None
    ) -> Dict[str, List[str]]:
        """
        Get comprehensive meal analysis including general tips and goal-specific advice.
        
        Args:
            dish_name: Name of the identified dish
            nutrition: Dictionary of nutritional values
            user_profile: Optional user profile for personalized analysis
            
        Returns:
            Dictionary with keys:
            - suggestions: List of improvement suggestions
            - tips: List of general healthy eating tips
            - goal_specific: List of goal-related advice
        """
        # Get basic suggestions
        suggestions = self.generate_suggestions(nutrition, user_profile)
        
        # Add general healthy eating tips
        tips = [
            "Remember to stay hydrated throughout the day 💧",
            "Take time to eat slowly and mindfully 🧘",
            "Include a variety of colors in your meals for different nutrients 🌈"
        ]
        
        # Add goal-specific advice if profile exists
        goal_specific = []
        if user_profile:
            if user_profile.goal == "bulking":
                goal_specific = [
                    "Aim for 5-6 meals per day",
                    "Include complex carbs for sustained energy",
                    "Consider post-workout protein shakes"
                ]
            elif user_profile.goal == "cutting":
                goal_specific = [
                    "Focus on protein to preserve muscle",
                    "Fill up on low-calorie vegetables",
                    "Stay mindful of portion sizes"
                ]
            else:  # maintenance
                goal_specific = [
                    "Maintain consistent meal timing",
                    "Balance your macronutrients",
                    "Listen to your hunger cues"
                ]
                
        return {
            "suggestions": suggestions,
            "tips": tips,
            "goal_specific": goal_specific
        }
