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
        """Run the vision model on the provided meal image and return a DishAnalysis.

        Improvements implemented here (method-local only):
        - EXIF orientation normalization & RGB conversion
        - Constrain both dimensions (<=512) using thumbnail
        - Optimized JPEG (quality 85, optimize, progressive) for smaller payload
        - Versioned cache key (allows future prompt/schema evolution)
        - More explicit, compact, rule-based prompt with examples & prohibitions
        - Defensive model response validation (empty choices/content)
        - Structured logging including hash, size, items count
        - Time stamp uses UTC without incorrect timezone attribute usage
        """
        try:
            # Import inside method to respect the 'do not change outside' requirement.
            try:
                from PIL import ImageOps  # type: ignore
            except Exception:  # pragma: no cover - optional enhancement
                ImageOps = None  # fallback if not available

            # 1. Normalize & copy
            img = image.copy()
            if ImageOps:
                try:
                    img = ImageOps.exif_transpose(img)
                except Exception:
                    pass
            if img.mode != "RGB":
                img = img.convert("RGB")

            # 2. Resize bounding box (maintain aspect ratio)
            max_dim = 512
            if img.width > max_dim or img.height > max_dim:
                img.thumbnail((max_dim, max_dim))

            # 3. Encode JPEG
            buf = io.BytesIO()
            try:
                img.save(buf, format="JPEG", quality=85, optimize=True, progressive=True)
            except Exception:
                # Fallback if optimize/progressive not supported
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
            data = buf.getvalue()
            img_b64 = base64.b64encode(data).decode()

            # 4. Versioned cache key (prompt/schema changes shouldn't reuse prior results)
            prompt_version = "v2"  # bump when materially changing prompt instructions
            base_hash = hashlib.md5(data).hexdigest()
            cache_key = f"{prompt_version}:{base_hash}:{self.VISION_MODEL}"
            if cache_key in self._vision_cache:
                return self._vision_cache[cache_key]

            # 5. Build compact schema & prompt
            schema = VisionAnalysisResponse.model_json_schema()
            # Minify schema to save tokens
            schema_json = json.dumps(schema, separators=(",", ":"))

            rules = (
                "Rules:\n"
                "1. Identify only visible, distinct food/drink items (no hidden/inferred ingredients).\n"
                "2. Count identical discrete items precisely (e.g., '3 tacos').\n"
                "3. Single dish in one plate/bowl => quantity '1 serving'.\n"
                "4. Multiple identical plates/bowls => 'N servings'.\n"
                "5. Confidence: 0.00–1.00 (two decimals). If unsure, use generic label + low confidence.\n"
                "6. No commentary, no markdown, no code fences — JSON ONLY.\n"
                "7. Do not list composite dish ingredients separately unless plated separately.\n"
                "8. If unsure of name, use 'unknown <category>' (e.g., 'unknown vegetable').\n"
            )

            examples = (
                "Examples:\n"
                "A: One plate spaghetti -> quantity '1 serving'.\n"
                "B: Three identical pancakes -> '3 pancakes'.\n"
                "C: Two bowls same soup -> '2 servings'.\n"
                "Bad (DO NOT): Extra text, markdown fences, inferred hidden ingredients.\n"
            )

            prompt = (
                "Analyze this meal image. Provide structured JSON only.\n\n" +
                rules + "\n" + examples +
                "Schema (all required):" + schema_json
            )

            # 6. Call model
            ts = datetime.utcnow().isoformat() + "Z"
            start = time.perf_counter()
            logger.info(
                "[OpenAI][START] ts=%s step=vision model=%s temp=%s max_tokens=%s w=%d h=%d bytes=%d hash=%s",
                ts, self.VISION_MODEL, self.JSON_TEMPERATURE, self.JSON_MAX_TOKENS, img.width, img.height, len(data), cache_key
            )

            resp = await self.client.chat.completions.create(
                model=self.VISION_MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are a precise food recognition AI. Return ONLY valid JSON; no markdown; comply strictly with the provided schema."
                    )},
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

            duration_ms = int((time.perf_counter() - start) * 1000)

            # 7. Defensive validation
            if not getattr(resp, "choices", None) or not resp.choices:
                raise ValueError("Empty response choices from vision model")
            first = resp.choices[0]
            content = getattr(getattr(first, "message", None), "content", None)
            if not content:
                raise ValueError("Empty message content from vision model")

            raw = self._clean_ai_response(content)
            parsed = VisionAnalysisResponse.model_validate_json(raw, strict=True)
            result = DishAnalysis(items=parsed.items)
            self._vision_cache[cache_key] = result

            logger.info(
                "[OpenAI][END] step=vision dur_ms=%d hash=%s items=%d", duration_ms, cache_key, len(result.items)
            )
            return result
        except Exception as e:
            logger.error("Error in _analyze_image: %s", e, exc_info=True)
            return DishAnalysis(items=[VisionFoodItem(name="Unidentified Food", type="main_dish", quantity="1 serving", confidence=0.0)])

    async def _analyze_nutrition(self, items: List[FoodItem]) -> NutritionAnalysis:
        """Analyze nutrition for a list of food items via LLM JSON response.

        Enhancements (method-local only):
        - Versioned cache key (separates prompt/schema evolution)
        - Compact minified schema & payload to reduce tokens
        - Explicit rule-based prompt with examples & prohibitions
        - Defensive response validation for empty choices/content
        - Structured logging (hash, item count, duration)
        - UTC timestamp suffix 'Z'
        Returns NutritionAnalysis or None on failure (unchanged externally).
        """
        try:
            # 1. Build normalized payload (stable ordering for cache key)
            payload = [
                {
                    "name": i.name,
                    "type": i.type,
                    "quantity": getattr(i, "quantity", "1 serving")
                }
                for i in items
            ]

            # 2. Versioned cache key (update version when prompt/schema materially changes)
            prompt_version = "nutri_v2"
            key_material = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            key_hash = hashlib.md5(key_material.encode()).hexdigest()
            cache_key = f"{prompt_version}:{key_hash}:{self.CHAT_MODEL}"
            if cache_key in self._nutrition_cache:
                return self._nutrition_cache[cache_key]

            # 3. Minified schema
            schema = NutritionAnalysisResponse.model_json_schema()
            schema_json = json.dumps(schema, separators=(",", ":"))

            # 4. Rules & examples (concise)
            rules = (
                "Rules:\n"
                "1. Use provided item names; do not invent new items.\n"
                "2. Estimate realistic serving-based nutrition macros & calories.\n"
                "3. Keep numeric units consistent (grams for macros, calories for energy).\n"
                "4. If uncertain, provide best estimate; never leave required fields blank.\n"
                "5. Do NOT add commentary, markdown, or extra keys. JSON ONLY.\n"
                "6. All required schema fields must be present.\n"
            )
            examples = (
                "Example item input -> output note: '2 tacos' should reflect combined nutrition for both tacos.\n"
                "If quantity has explicit count (e.g., '3 pancakes'), scale nutrition accordingly.\n"
            )

            # 5. Prompt assembly (payload inline, minified)
            prompt = (
                "Analyze these meal components and return ONLY valid JSON matching the schema.\n" +
                rules + examples +
                "Schema:" + schema_json + "\nItems:" + key_material
            )

            ts = datetime.utcnow().isoformat() + "Z"
            start = time.perf_counter()
            logger.info(
                "[OpenAI][START] ts=%s step=nutrition model=%s temp=%s max_tokens=%s items=%d hash=%s",
                ts, self.CHAT_MODEL, self.JSON_TEMPERATURE, self.JSON_MAX_TOKENS, len(payload), cache_key
            )

            resp = await self.client.chat.completions.create(
                model=self.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a meticulous nutrition scientist. Return ONLY JSON conforming to the schema."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=self.JSON_TEMPERATURE,
                max_tokens=self.JSON_MAX_TOKENS,
            )

            duration_ms = int((time.perf_counter() - start) * 1000)

            # Defensive checks
            if not getattr(resp, "choices", None) or not resp.choices:
                raise ValueError("Empty response choices from nutrition model")
            first = resp.choices[0]
            content = getattr(getattr(first, "message", None), "content", None)
            if not content:
                raise ValueError("Empty message content from nutrition model")

            raw = self._clean_ai_response(content)
            parsed = NutritionAnalysisResponse.model_validate_json(raw)
            result = parsed.to_nutrition_analysis()
            self._nutrition_cache[cache_key] = result

            logger.info(
                "[OpenAI][END] step=nutrition dur_ms=%d hash=%s items=%d", duration_ms, cache_key, len(result.items if hasattr(result, 'items') else [])
            )
            return result
        except Exception as e:
            logger.error("Error in _analyze_nutrition: %s", e, exc_info=True)
            return None
    async def _get_recommendations(self, nutrition: Optional[Any]) -> Dict[str, Any]:
        """Generate meal recommendations and qualitative feedback.

        Enhancements (method-local only):
        - Rule-based prompt with explicit output contract
        - Versioned cache (to avoid recomputation for identical macro profiles)
        - Defensive validation of model response
        - Structured logging with hash, duration, and macro summary
        - Consistent UTC timestamp with 'Z'
        Caches by a hash of macro summary + prompt version.
        """
        try:
            # 1. Normalize nutrition summary into a simple object (calories, macros)
            if nutrition and hasattr(nutrition, "combined"):
                combined = nutrition.combined
            elif isinstance(nutrition, dict) and "nutrition_summary" in nutrition:
                ns = nutrition["nutrition_summary"]
                combined = type("obj", (), {
                    "calories": ns.get("calories", {}).get("value", 0),
                    "protein_g": ns.get("macros", {}).get("protein", {}).get("value", 0),
                    "carbs_g": ns.get("macros", {}).get("carbs", {}).get("value", 0),
                    "fat_g": ns.get("macros", {}).get("fat", {}).get("value", 0),
                })()
            else:
                combined = type("obj", (), {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0})()

            meal_summary = (
                f"Calories {combined.calories} kcal, Protein {combined.protein_g}g, "
                f"Carbs {combined.carbs_g}g, Fat {combined.fat_g}g"
            )

            # 2. Versioned cache key for recommendations (coarse granularity by macro quartet)
            prompt_version = "rec_v2"
            key_material = json.dumps({
                "calories": combined.calories,
                "protein_g": combined.protein_g,
                "carbs_g": combined.carbs_g,
                "fat_g": combined.fat_g,
                "v": prompt_version,
            }, sort_keys=True, separators=(",", ":"))
            rec_hash = hashlib.md5(key_material.encode()).hexdigest()
            cache_key = f"{prompt_version}:{rec_hash}:{self.CHAT_MODEL}"
            # Initialize recommendation cache lazily
            if not hasattr(self, "_rec_cache"):
                self._rec_cache = {}
            if cache_key in self._rec_cache:
                return self._rec_cache[cache_key]

            # 3. Rules & JSON contract (concise)
            contract = (
                "Required JSON keys: recommendations[list(str)], health_score[int 0-100], meal_type[str one of "
                "breakfast|lunch|dinner|snack|unknown], dietary_considerations[list(str)], meal_rating[int 0-10], "
                "suggestions[list(str)], improvements[list(str)], positive_aspects[list(str)]."
            )
            rules = (
                "Rules:\n"
                "1. Base feedback ONLY on provided macros (no guessing hidden nutrients).\n"
                "2. health_score: holistic quality (higher = healthier).\n"
                "3. meal_rating: palatability/overall quality 0–10.\n"
                "4. Provide actionable, concise, non-repetitive suggestions.\n"
                "5. No markdown, explanations, or extra keys — JSON ONLY.\n"
                "6. All arrays must have at least 1 element; if nothing relevant, give a neutral constructive entry.\n"
            )
            examples = (
                "Example health_score guidance: High protein & moderate calories => 70–85; very high sugar & low protein => 30–50.\n"
            )

            prompt = (
                f"Macro summary: {meal_summary}\n" +
                contract + "\n" + rules + examples +
                "Return ONLY strict JSON with all required keys." 
            )

            ts = datetime.utcnow().isoformat() + "Z"
            start = time.perf_counter()
            logger.info(
                "[OpenAI][START] ts=%s step=recommendations model=%s temp=%s max_tokens=%s hash=%s summary='%s'",
                ts, self.CHAT_MODEL, self.JSON_TEMPERATURE, self.JSON_MAX_TOKENS, rec_hash, meal_summary
            )

            resp = await self.client.chat.completions.create(
                model=self.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a concise, evidence-based nutrition coach. Output ONLY JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=self.JSON_TEMPERATURE,
                max_tokens=self.JSON_MAX_TOKENS,
            )

            duration_ms = int((time.perf_counter() - start) * 1000)

            if not getattr(resp, "choices", None) or not resp.choices:
                raise ValueError("Empty response choices from recommendations model")
            first = resp.choices[0]
            content = getattr(getattr(first, "message", None), "content", None)
            if not content:
                raise ValueError("Empty message content from recommendations model")

            raw = self._clean_ai_response(content)
            parsed = MealRecommendations.model_validate_json(raw)
            result = parsed.model_dump()

            # Clamp numeric fields and enforce non-empty arrays
            result["health_score"] = max(0, min(100, int(result.get("health_score", 0) or 0)))
            result["meal_rating"] = max(0, min(10, int(result.get("meal_rating", 0) or 0)))
            for arr_key, fallback in [
                ("recommendations", "General balanced meal advice."),
                ("dietary_considerations", "none"),
                ("suggestions", "Add a source of fiber."),
                ("improvements", "Increase vegetables."),
                ("positive_aspects", "Contains protein."),
            ]:
                if not isinstance(result.get(arr_key), list) or len(result.get(arr_key)) == 0:
                    result[arr_key] = [fallback]

            # Cache result
            self._rec_cache[cache_key] = result

            logger.info(
                "[OpenAI][END] step=recommendations dur_ms=%d hash=%s recs=%d", duration_ms, rec_hash, len(result.get('recommendations', []))
            )
            return result
        except Exception as e:
            logger.error("Error in _get_recommendations: %s", e, exc_info=True)
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
        """Answer user questions about the analyzed meal with a structured streaming response.

        Enhancements:
        - Structured system instructions with explicit output contract
        - Uses delimiters (---SUMMARY--- / ---DETAILS---) for easier client-side parsing
        - Wraps OpenAI streaming iterator into an async generator yielding text chunks
        - Logs start/end with duration and char count
        - Defensive handling for empty/None content events
        Returns: async generator of str chunks (original external behavior preserved if caller consumes returned stream)
        """
        try:
            nutrition = meal_data.get("nutrition_summary", {})
            meal_name = meal_data.get('meal_info', {}).get('name', 'Unknown')

            def fmt_entry(k: str, v: Dict[str, Any]) -> str:
                try:
                    val = v.get('value')
                    if val is None:
                        return ''
                    if k == 'calories':
                        return f"calories: {round(val)} kcal"
                    return f"{k}: {round(val)}g"
                except Exception:
                    return ''

            nutrition_parts = [fmt_entry(k, v) for k, v in nutrition.items() if isinstance(v, dict)]
            nutrition_text = ", ".join([p for p in nutrition_parts if p]) or "no macro data"

            system_msg = (
                "You are an evidence-based registered dietitian. Provide accurate, concise, user-friendly guidance.\n"
                "Format strictly with two sections delimited by markers.\n"
                "OUTPUT CONTRACT:\n"
                "---SUMMARY---\n"
                "One or two punchy sentences answering the question. May include tasteful emojis (max 2).\n"
                "---DETAILS---\n"
                "Bullet-style or short paragraphs: context, nutrient implications, actionable advice, daily value references.\n"
                "RULES:\n"
                "1. No markdown headings besides the required markers.\n"
                "2. Avoid medical claims; focus on general nutrition guidance.\n"
                "3. Be encouraging, specific, not alarmist.\n"
                "4. If data insufficient, state assumptions explicitly.\n"
                "5. Never fabricate exact micronutrient numbers not provided.\n"
                "6. Keep DETAILS under ~12 sentences total.\n"
            )

            user_msg = (
                f"Meal: {meal_name}. Nutrition summary: {nutrition_text}.\n"
                f"Question: {question}"
            )

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]

            ts = datetime.utcnow().isoformat() + "Z"
            start = time.perf_counter()
            logger.info(
                "[OpenAI][START] ts=%s step=qa_stream model=%s meal=%s question_len=%d", ts, self.CHAT_MODEL, meal_name, len(question)
            )

            stream = await self.client.chat.completions.create(
                model=self.CHAT_MODEL,
                messages=messages,
                stream=True,
                max_tokens=500,
                temperature=0.5,
            )

            async def generator():
                emitted = 0
                try:
                    async for chunk in stream:  # type: ignore
                        # OpenAI Python async streaming yields events with .choices[0].delta.content sometimes None
                        try:
                            choices = getattr(chunk, 'choices', None)
                            if not choices:
                                continue
                            delta = getattr(choices[0], 'delta', None)
                            if not delta:
                                continue
                            content = getattr(delta, 'content', None)
                            if content:
                                emitted += len(content)
                                yield content
                        except Exception as inner_e:  # continue streaming despite small parse errors
                            logger.debug("[Stream][WARN] parse_error=%s", inner_e)
                            continue
                finally:
                    dur_ms = int((time.perf_counter() - start) * 1000)
                    logger.info(
                        "[OpenAI][END] step=qa_stream dur_ms=%d chars=%d meal=%s", dur_ms, emitted, meal_name
                    )

            return generator()
        except Exception as e:
            logger.error("Error in answer_question: %s", e, exc_info=True)
            raise
