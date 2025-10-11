import json
import logging
import hashlib
import asyncio
import time
import base64
import io
from datetime import datetime
from typing import Dict, List, Optional, Any
from PIL import Image
from openai import AsyncOpenAI

from app.models import (
    DishAnalysis,
    VisionFoodItem,
    NutritionInfo,
    NutritionAnalysis,
    VisionAnalysisResponse,
    NutritionAnalysisResponse,
    MealRecommendations,
    FoodItem,
    FoodItemWithNutrition,
)
from app.config import VISION_MODEL, CHAT_MODEL

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _f = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    _h.setFormatter(_f)
    logger.addHandler(_h)
logger.setLevel(logging.INFO)
logger.propagate = False


class FoodAnalyzerService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = AsyncOpenAI(api_key=api_key)
        self.VISION_MODEL = VISION_MODEL
        self.CHAT_MODEL = CHAT_MODEL
        self.JSON_TEMPERATURE = 0.1
        self.JSON_MAX_TOKENS = 1500
        self.TEXT_TEMPERATURE = 0.3
        self.TEXT_MAX_TOKENS = 400
        self._vision_cache = {}
        self._nutrition_cache = {}

    def _clean_ai_response(self, text: str) -> str:
        if not text:
            return "{}"
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`")
            t = t.replace("json", "").strip()
        return t

    async def analyze_meal(self, image: Image.Image) -> Dict[str, Any]:
        try:
            scene_analysis = await self._analyze_image(image)
            if not scene_analysis or not scene_analysis.items:
                return {"status": "error", "error": "No food detected"}

            nutrition_task = asyncio.create_task(self._analyze_nutrition(scene_analysis.items))
            recommendations_task = asyncio.create_task(self._get_recommendations(None))
            nutrition, recommendations = await asyncio.gather(nutrition_task, recommendations_task)

            if not nutrition:
                return {"status": "error", "error": "Nutrition analysis failed"}

            if not recommendations:
                recommendations = {
                    "recommendations": ["No specific suggestions"],
                    "health_score": 5,
                    "meal_type": "unknown",
                    "dietary_considerations": [],
                    "meal_rating": 5,
                    "suggestions": [],
                    "improvements": [],
                    "positive_aspects": [],
                }

            return {
                "status": "success",
                "meal_info": {
                    "name": getattr(scene_analysis.items[0], "name", "Unknown Dish"),
                    "confidence": getattr(scene_analysis.items[0], "confidence", 0.0),
                    "serving_size": nutrition.combined.serving_size,
                },
                "nutrition_summary": {
                    "calories": {
                        "value": nutrition.combined.calories,
                        "daily_value": int((nutrition.combined.calories / 2000) * 100),
                    },
                    "macros": {
                        "protein": {
                            "value": nutrition.combined.protein_g,
                            "daily_value": int((nutrition.combined.protein_g / 50) * 100),
                        },
                        "carbs": {
                            "value": nutrition.combined.carbs_g,
                            "daily_value": int((nutrition.combined.carbs_g / 225) * 100),
                        },
                        "fat": {
                            "value": nutrition.combined.fat_g,
                            "daily_value": int((nutrition.combined.fat_g / 65) * 100),
                        },
                    },
                    "additional": {
                        "fiber": {
                            "value": nutrition.combined.fiber_g,
                            "daily_value": int((nutrition.combined.fiber_g / 28) * 100),
                        },
                        "sugar": {
                            "value": nutrition.combined.sugar_g,
                            "warning": nutrition.combined.sugar_g > 20,
                        },
                        "saturated_fat": {
                            "value": nutrition.combined.saturated_fat_g,
                            "daily_value": int((nutrition.combined.saturated_fat_g / 20) * 100),
                        },
                    },
                    "diet_tags": nutrition.combined.diet_tags,
                    "warnings": nutrition.combined.warnings,
                },
                "components": [
                    {
                        "name": i.name,
                        "type": i.type,
                        "serving": i.nutrition.serving_size,
                        "nutrition": {
                            "calories": i.nutrition.calories,
                            "protein": i.nutrition.protein_g,
                            "carbs": i.nutrition.carbs_g,
                            "fat": i.nutrition.fat_g,
                            "fiber": i.nutrition.fiber_g,
                            "sugar": i.nutrition.sugar_g,
                            "saturated_fat": i.nutrition.saturated_fat_g,
                        },
                        "diet_tags": i.nutrition.diet_tags,
                    }
                    for i in nutrition.items
                ],
                "ai_insights": recommendations,
            }
        except Exception as e:
            logger.error(f"Error analyzing meal: {e}")
            return {"status": "error", "error": str(e)}

    async def _analyze_image(self, image: Image.Image) -> DishAnalysis:
        try:
            img = image.copy()
            if img.width > 512:
                ratio = 512 / img.width
                img = img.resize((512, int(img.height * ratio)))

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            data = buf.getvalue()
            img_b64 = base64.b64encode(data).decode()
            img_hash = hashlib.md5(data).hexdigest()

            if img_hash in self._vision_cache:
                return self._vision_cache[img_hash]

            schema = VisionAnalysisResponse.model_json_schema()
            prompt = f"""Analyze the meal image and identify all food items visible.

IMPORTANT: Carefully count the number of servings/portions shown in the image:
- If you see 1 plate/bowl/container, set quantity to "1 serving"
- If you see 2 plates/bowls/containers with the same food, set quantity to "2 servings"
- If you see multiple portions (e.g., 2 sandwiches, 3 tacos), include the count in quantity (e.g., "2 sandwiches", "3 tacos")

Return valid JSON following this schema:
{json.dumps(schema, indent=2)}"""

            _t = datetime.utcnow().isoformat()
            _s = time.perf_counter()
            logger.info(f"[OpenAI][START] ts={_t} step=vision model={self.VISION_MODEL} temp={self.JSON_TEMPERATURE} max_tokens={self.JSON_MAX_TOKENS} img_w={img.width} img_h={img.height}")

            resp = await self.client.chat.completions.create(
                model=self.VISION_MODEL,
                messages=[
                    {"role": "system", "content": "You are a precise food recognition AI that accurately counts servings and portions. Return only valid JSON."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        ],
                    },
                ],
                response_format={"type": "json_object"},
                temperature=self.JSON_TEMPERATURE,
                max_tokens=self.JSON_MAX_TOKENS,
            )

            dur = int((time.perf_counter() - _s) * 1000)
            logger.info(f"[OpenAI][END] step=vision dur_ms={dur}")

            raw = self._clean_ai_response(resp.choices[0].message.content)
            parsed = VisionAnalysisResponse.model_validate_json(raw, strict=True)
            result = DishAnalysis(items=parsed.items)
            self._vision_cache[img_hash] = result
            return result
        except Exception as e:
            logger.error(f"Error in _analyze_image: {e}")
            return DishAnalysis(items=[VisionFoodItem(name="Unidentified Food", type="main_dish", quantity="1 serving", confidence=0.0)])

    async def _analyze_nutrition(self, items: List[FoodItem]) -> NutritionAnalysis:

        try:
            payload = [{"name": i.name, "type": i.type, "quantity": getattr(i, "quantity", "1 serving")} for i in items]
            key = json.dumps(payload, sort_keys=True)
            if key in self._nutrition_cache:
                return self._nutrition_cache[key]

            prompt = f"""
            Analyze these items and return precise nutritional info:
            {json.dumps(payload, indent=2)}
            JSON schema: {json.dumps(NutritionAnalysisResponse.model_json_schema(), indent=2)}
            """

            _t = datetime.utcnow().isoformat()
            _s = time.perf_counter()
            logger.info(f"[OpenAI][START] ts={_t} step=nutrition model={self.CHAT_MODEL} temp=0.1 max_tokens=1500")

            resp = await self.client.chat.completions.create(
                model=self.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a nutrition scientist. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=self.JSON_TEMPERATURE,
                max_tokens=self.JSON_MAX_TOKENS,
            )

            dur = int((time.perf_counter() - _s) * 1000)
            logger.info(f"[OpenAI][END] step=nutrition dur_ms={dur}")

            raw = self._clean_ai_response(resp.choices[0].message.content)
            parsed = NutritionAnalysisResponse.model_validate_json(raw)
            result = parsed.to_nutrition_analysis()
            self._nutrition_cache[key] = result
            return result
        except Exception as e:
            logger.error(f"Error in _analyze_nutrition: {e}")
            return None
    async def _get_recommendations(self, nutrition: Optional[Any]) -> Dict[str, Any]:
        try:
            sys = "You are a nutrition coach giving concise, factual meal feedback."

            # Safely handle dict or NutritionAnalysis
            if nutrition and hasattr(nutrition, "combined"):
                combined = nutrition.combined
            elif isinstance(nutrition, dict) and "nutrition_summary" in nutrition:
                ns = nutrition["nutrition_summary"]
                combined = type("obj", (), {
                    "calories": ns["calories"]["value"],
                    "protein_g": ns["macros"]["protein"]["value"],
                    "carbs_g": ns["macros"]["carbs"]["value"],
                    "fat_g": ns["macros"]["fat"]["value"],
                })()
            else:
                combined = type("obj", (), {
                    "calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0
                })()

            meal_summary = (
                f"Calories {combined.calories} kcal, "
                f"Protein {combined.protein_g}g, "
                f"Carbs {combined.carbs_g}g, "
                f"Fat {combined.fat_g}g"
            )

            prompt = f"""
            Based on: {meal_summary}
            Return valid JSON including **all** the following fields:

            {{
              "recommendations": ["Tip 1", "Tip 2"],
              "health_score": int(0-100),
              "meal_type": "breakfast|lunch|dinner|snack|unknown",
              "dietary_considerations": ["list"],
              "meal_rating": int(0-10),
              "suggestions": ["Quick suggestion 1", "Quick suggestion 2"],
              "improvements": ["Improvement idea 1", "Improvement idea 2"],
              "positive_aspects": ["Positive note 1", "Positive note 2"]
            }}

            Ensure **meal_rating** is a numeric score (0–10).
            Always include all keys. Return only JSON.
            """

            _t = datetime.utcnow().isoformat()
            _s = time.perf_counter()
            logger.info(f"[OpenAI][START] ts={_t} step=recommendations model={self.CHAT_MODEL} temp=0.1 max_tokens=1500")

            resp = await self.client.chat.completions.create(
                model=self.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=self.JSON_TEMPERATURE,
                max_tokens=self.JSON_MAX_TOKENS,
            )

            dur = int((time.perf_counter() - _s) * 1000)
            logger.info(f"[OpenAI][END] step=recommendations dur_ms={dur}")

            raw = self._clean_ai_response(resp.choices[0].message.content)
            parsed = MealRecommendations.model_validate_json(raw)
            return parsed.model_dump()
        except Exception as e:
            logger.error(f"Error in _get_recommendations: {e}")
            return {
                "recommendations": ["Could not generate advice."],
                "health_score": 5,
                "meal_type": "unknown",
                "dietary_considerations": [],
                "meal_rating": 5,
                "suggestions": [],
                "improvements": [],
                "positive_aspects": [],
            }

    async def suggest_improvements(self, nutrition: NutritionAnalysis) -> List[str]:
        """Public helper to get quick improvement suggestions."""
        try:
            rec = await self._get_recommendations(nutrition)
            return rec.get("improvements", []) or rec.get("recommendations", [])
        except Exception as e:
            logger.error(f"Error in suggest_improvements: {e}")
            return ["No improvements available."]

    async def answer_question(self, question: str, meal_data: Dict[str, Any]):
        """Answer user questions about the analyzed meal with structured streaming response."""
        try:
            nutrition = meal_data.get("nutrition_summary", {})
            nutrition_text = ", ".join(
                f"{k}: {round(v['value'])}{' kcal' if k == 'calories' else 'g'}"
                for k, v in nutrition.items()
                if isinstance(v, dict) and "value" in v
            )

            system_msg = (
                "You are an expert nutritionist and registered dietitian. "
                "Respond in two sections:\n\n"
                "💡 SUMMARY:\n"
                "- Give a short, direct answer in 1–2 sentences with emojis.\n\n"
                "📘 DETAILS:\n"
                "- Explain nutritional relevance.\n"
                "- Reference daily values and context.\n"
                "- Give practical, actionable advice."
            )

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"The meal is {meal_data.get('meal_info', {}).get('name', 'Unknown')}. "
                                            f"Nutrition: {nutrition_text}. "
                                            f"Question: {question}"}
            ]

            logger.info(f"[OpenAI][START] ts={datetime.utcnow().isoformat()} step=qa_stream model={self.CHAT_MODEL}")

            stream = await self.client.chat.completions.create(
                model=self.CHAT_MODEL,
                messages=messages,
                stream=True,
                max_tokens=500,
                temperature=0.5,
            )

            return stream

        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            raise
