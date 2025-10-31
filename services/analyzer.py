import json
import logging
import hashlib
import asyncio
import time
import base64
import io
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Sequence, cast
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
    FoodItemBase,
    FoodItemWithNutrition,
)
from app.config import (
    VISION_MODEL, 
    CHAT_MODEL, 
    JSON_TEMPERATURE,
    JSON_MAX_TOKENS,
    JSON_MAX_TOKENS_NUTRITION,
    TEXT_TEMPERATURE,
    TEXT_MAX_TOKENS,
    FAST_MODE,
    MAX_FOOD_ITEMS
)

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

def _get_user_context(user_preferences: Optional[dict] = None) -> str:
    # 3. Build user context from preferences
    user_context = ""
    if user_preferences:
        goal = user_preferences.get("goal", "Balanced")
        diet = user_preferences.get("diet", "None")
        health_conditions = user_preferences.get("health_conditions", [])
        calorie_target = user_preferences.get("calorie_target", 2500)

        user_context = (
            f"\nUSER PROFILE:\n"
            f"- Goal: {goal}\n"
            f"- Diet: {diet}\n"
            f"- Calorie Target: {calorie_target} kcal/day\n"
        )
        if health_conditions:
            user_context += f"- Health Conditions: {', '.join(health_conditions)}\n"

        # ---- GOAL GUIDANCE ----
        goal_guidance_map = {
            "Cutting": "Cutting goal: prioritize satiety and adequate protein; avoid calorie increases.",
            "Bulking": "Bulking goal: ensure sufficient calories and high protein for muscle growth.",
            "Maintenance": "Maintenance goal: maintain balanced, sustainable nutrition.",
            "Balanced": "Balanced goal: general health focus with even macro distribution.",
        }
        goal_guidance = goal_guidance_map.get(goal, "Balanced goal: general health focus.")

        # ---- DIET GUIDANCE ----
        diet_guidance_map = {
            "Vegan": "Vegan diet: plant-based proteins (tofu, lentils), B12 and iron sources, calcium from greens or tahini.",
            "Vegetarian": "Vegetarian diet: plant proteins and dairy; ensure B12 and iron intake.",
            "Keto": "Keto diet: low-carb, high-fat, moderate protein; avoid sugars and grains.",
            "Mediterranean": "Mediterranean diet: olive oil, fish, nuts, legumes, fresh vegetables.",
        }
        diet_guidance = diet_guidance_map.get(
            diet, f"{diet} diet: apply appropriate dietary logic." if diet != "None" else "No specific diet."
        )

        # ---- HEALTH GUIDANCE ----
        health_guidance = ""
        if health_conditions:
            health_guidance = "Health Conditions:\n"
            for cond in health_conditions:
                c = cond.lower()
                if "diabetes" in c:
                    health_guidance += "- Diabetes: prefer low-GI, fiber-rich, balanced carb-protein meals.\n"
                elif "hypertension" in c or "high blood pressure" in c:
                    health_guidance += "- Hypertension: low sodium, potassium-rich, heart-healthy foods.\n"
                elif "heart" in c or "cardio" in c:
                    health_guidance += "- Heart Disease: omega-3s, fiber, low saturated fat.\n"
                elif "cholesterol" in c:
                    health_guidance += "- High Cholesterol: soluble fiber, lean proteins, avoid trans fats.\n"
                elif "kidney" in c or "renal" in c:
                    health_guidance += "- Kidney Disease: moderate protein, low sodium and phosphorus.\n"
                elif "celiac" in c or "gluten" in c:
                    health_guidance += "- Celiac/Gluten Sensitivity: ensure gluten-free foods.\n"
                elif "ibs" in c:
                    health_guidance += "- IBS: low-FODMAP, easily digestible foods, watch triggers.\n"
                elif "lactose" in c:
                    health_guidance += "- Lactose Intolerance: use lactose-free or calcium-rich non-dairy options.\n"
                else:
                    health_guidance += f"- {cond}: apply standard dietary precautions.\n"

        user_context += f"{goal_guidance}\n{diet_guidance}\n{health_guidance}"
        user_context += (
            "Important: Always include diet- or condition-specific suggestions. "
            "Never return 'no improvements needed' if user has active preferences or conditions.\n"
        )
    return user_context


class FoodAnalyzerService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = AsyncOpenAI(api_key=api_key)
        self.VISION_MODEL = VISION_MODEL
        self.CHAT_MODEL = CHAT_MODEL
        self.JSON_TEMPERATURE = JSON_TEMPERATURE
        self.JSON_MAX_TOKENS = JSON_MAX_TOKENS
        self.JSON_MAX_TOKENS_NUTRITION = JSON_MAX_TOKENS_NUTRITION
        self.TEXT_TEMPERATURE = TEXT_TEMPERATURE
        self.TEXT_MAX_TOKENS = TEXT_MAX_TOKENS
        self.FAST_MODE = FAST_MODE
        self.MAX_FOOD_ITEMS = MAX_FOOD_ITEMS
        self._vision_cache = {}
        self._nutrition_cache = {}
    
    def enable_ultra_fast_mode(self):
        """Enable ultra-fast mode for even faster analysis."""
        self.FAST_MODE = True
        self.MAX_FOOD_ITEMS = 5
        self.JSON_MAX_TOKENS_NUTRITION = 500
        self.JSON_MAX_TOKENS = 700
        logger.info("[SPEED] Ultra-fast mode enabled: 5 items max, reduced tokens")

    def _clean_ai_response(self, text: str) -> str:
        if not text:
            return "{}"
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`")
            t = t.replace("json", "").strip()
        return t

    async def analyze_meal(self, image: Image.Image, user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        start_time = time.perf_counter()
        try:
            logger.info("[MEAL_ANALYSIS][START] Starting optimized meal analysis")
            
            vision_start = time.perf_counter()
            scene_analysis = await self._analyze_image(image)
            vision_time = int((time.perf_counter() - vision_start) * 1000)
            logger.info(f"[MEAL_ANALYSIS][VISION] Completed in {vision_time}ms")
            
            if not scene_analysis or not scene_analysis.items:
                return {"status": "error", "error": "No food detected"}

            # Speed optimization: limit food items if in fast mode
            items_to_analyze = scene_analysis.items
            if self.FAST_MODE and len(items_to_analyze) > self.MAX_FOOD_ITEMS:
                # Keep only the highest confidence items
                items_to_analyze = sorted(items_to_analyze, key=lambda x: x.confidence, reverse=True)[:self.MAX_FOOD_ITEMS]
                logger.info(f"[SPEED_OPTIMIZATION] Limited to {self.MAX_FOOD_ITEMS} highest confidence items (was {len(scene_analysis.items)})")

            # Start nutrition analysis immediately after vision
            nutrition_start = time.perf_counter()
            nutrition_task = asyncio.create_task(self._analyze_nutrition(items_to_analyze))
            nutrition = await nutrition_task
            nutrition_time = int((time.perf_counter() - nutrition_start) * 1000)
            logger.info(f"[MEAL_ANALYSIS][NUTRITION] Completed in {nutrition_time}ms")

            if not nutrition:
                return {"status": "error", "error": "Nutrition analysis failed"}

            result = {
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
                # AI insights will be generated separately via suggest_improvements()
            }
            
            total_time = int((time.perf_counter() - start_time) * 1000)
            logger.info(f"[MEAL_ANALYSIS][COMPLETE] Total analysis time: {total_time}ms")
            return result
            
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
            assert img is not None, "Failed to copy image"
            img = cast(Image.Image, img)  # Tell type checker img is not None
            if ImageOps:
                try:
                    img = ImageOps.exif_transpose(img)
                    assert img is not None, "Failed to transpose image"
                    img = cast(Image.Image, img)  # Tell type checker img is not None after transpose
                except Exception:
                    pass
            if img.mode != "RGB":  # type: ignore
                img = img.convert("RGB")  # type: ignore

            # 2. Resize bounding box (maintain aspect ratio)
            max_dim = 512
            if img.width > max_dim or img.height > max_dim:  # type: ignore
                img.thumbnail((max_dim, max_dim))  # type: ignore

            # 3. Encode JPEG
            buf = io.BytesIO()
            try:
                img.save(buf, format="JPEG", quality=85, optimize=True, progressive=True)  # type: ignore
            except Exception:
                # Fallback if optimize/progressive not supported
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)  # type: ignore
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
    "1. Identify all distinct visible food or drink items separately (do not group as 'salad bowl').\n"
    "2. Include solid foods, liquids, and condiments if clearly visible.\n"
    "3. For each item, include an estimated quantity (piece count, half portion, or grams).\n"
    "4. Always include an approximate weight in grams for visible servings, even if only one piece (e.g., '1 piece ≈120 g').\n"
    "5. Use real-world serving approximations: e.g., 1 egg, ½ avocado, 50 g rice, 40 g beans.\n"
    "6. Confidence: 0.00–1.00 (two decimals).\n"
    "7. Return only a JSON list named 'items' — no text, no markdown, no comments.\n"
    "8. For uncertain quantities, provide a low confidence and approximate descriptor (e.g., 'about 40 g').\n"
    "9. Label unknowns as 'unknown <category>' if the class cannot be identified.\n"
    "10. Do not infer hidden ingredients (e.g., filling) unless clearly visible.\n"
    "11. If a food visually matches a well-known regional or common dish (e.g., pizza, knafeh) "
    "with ≥ 0.7 confidence, name the dish directly instead of generic terms, but include the confidence score.\n"
)


            examples = (
                "Examples:\n"
                "A: One bowl containing chicken, rice, avocado, egg, and beans ->\n"
                "{'items':[{'name':'grilled chicken breast','quantity':'80 g','confidence':0.95},"
                "{'name':'brown rice','quantity':'50 g','confidence':0.90},"
                "{'name':'black beans','quantity':'40 g','confidence':0.92},"
                "{'name':'boiled egg','quantity':'1 piece','confidence':0.98},"
                "{'name':'avocado','quantity':'0.5 piece','confidence':0.95},"
                "{'name':'mixed greens','quantity':'60 g','confidence':0.90},"
                "{'name':'vinaigrette dressing','quantity':'20 g','confidence':0.85}]}\n"
                "B: Water glass with lemon slice -> {'items':[{'name':'lemon water','quantity':'1 glass','confidence':0.95}]}\n"
                "C: Red apple -> {'items':[{'name':'apple','quantity':'1 piece','confidence':0.98}]}\n"
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
                ts, self.VISION_MODEL, self.JSON_TEMPERATURE, self.JSON_MAX_TOKENS, img.width, img.height, len(data), cache_key  # type: ignore
            )

            try:
                resp = await asyncio.wait_for(
                    self.client.chat.completions.create(
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
                    ),
                    timeout=60.0  # Increased timeout to 60 seconds for vision analysis
                )
            except asyncio.TimeoutError:
                logger.error("OpenAI vision analysis timed out after 60 seconds")
                raise Exception("Vision analysis timed out - this may be due to network issues or API overload")
            except Exception as e:
                logger.error("OpenAI vision analysis failed: %s", e)
                raise Exception(f"Vision analysis failed: {str(e)}")

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
            
            # Check if any items are of type 'unknown' and handle accordingly
            has_unknown = any(item.type == "unknown" for item in result.items)
            if has_unknown:
                logger.warning("Non-food or unknown items detected in image")
                # Return a clear indication that this is not a food item
                return DishAnalysis(items=[VisionFoodItem(name="Non-Food Item Detected", type="unknown", quantity="N/A", confidence=0.0)])
            
            self._vision_cache[cache_key] = result

            logger.info(
                "[OpenAI][END] step=vision dur_ms=%d hash=%s items=%d", duration_ms, cache_key, len(result.items)
            )
            return result
        except Exception as e:
            logger.error("Error in _analyze_image: %s", e, exc_info=True)
            return DishAnalysis(items=[VisionFoodItem(name="Unidentified Food", type="unknown", quantity="N/A", confidence=0.0)])

    async def _analyze_nutrition(self, items: Sequence[FoodItemBase]) -> Optional[NutritionAnalysis]:
        try:
            # Check if we have any unknown food types (non-food items)
            if any(item.type == "unknown" for item in items):
                logger.info("Skipping nutrition analysis for unknown/non-food items")
                # Return empty nutrition data for non-food items
                empty_nutrition_info = NutritionInfo(
                    calories=0,
                    serving_size="N/A",
                    protein_g=0.0,
                    carbs_g=0.0,
                    fiber_g=0.0,
                    sugar_g=0.0,
                    fat_g=0.0,
                    saturated_fat_g=0.0,
                    category="N/A",
                    diet_tags=[],
                    warnings=["This appears to be a non-food item"]
                )
                empty_food_item = FoodItemWithNutrition(
                    name="Non-Food Item",
                    type="unknown",
                    nutrition=empty_nutrition_info
                )
                return NutritionAnalysis(
                    combined=empty_nutrition_info,
                    items=[empty_food_item]
                )

            # 1. Build normalized payload
            payload = [
                {
                    "name": i.name,
                    "type": i.type,
                    "quantity": getattr(i, "quantity", "1 serving")
                }
                for i in items
            ]

            # 2. Versioned cache key
            prompt_version = "nutri_v3"  # bumped version after schema enforcement change
            key_material = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            key_hash = hashlib.md5(key_material.encode()).hexdigest()
            cache_key = f"{prompt_version}:{key_hash}:{self.CHAT_MODEL}"
            if cache_key in self._nutrition_cache:
                return self._nutrition_cache[cache_key]

            # 3. Get minified schema
            schema_dict = NutritionAnalysisResponse.model_json_schema()
            schema_json = json.dumps(schema_dict, separators=(",", ":"))

            # 4. Rules & examples (optimized for speed in fast mode)
            if self.FAST_MODE:
                rules = (
                    "Rules: 1) Use exact item names/quantities 2) Estimate realistic nutrition "
                    "3) All required fields: Calories, Serving_Size, Protein, Carbs, Fiber, Sugar, Fat, Sat_Fat, Category "
                    "4) Return ONLY JSON, no markdown 5) Round to 1 decimal\n"
                )
                examples = (
                    "Example: Input:[{'name':'Salad','type':'side_dish','quantity':'1 bowl'}] "
                    "Output:{'combined':{'Calories':150,'Serving_Size':'1 bowl','Protein':5.0,'Carbs':20.0,'Fiber':8.0,'Sugar':10.0,'Fat':8.0,'Sat_Fat':2.0,'Category':'salad'},"
                    "'items':[{'name':'Salad','type':'side_dish','Calories':150,'Serving_Size':'1 bowl','Protein':5.0,'Carbs':20.0,'Fiber':8.0,'Sugar':10.0,'Fat':8.0,'Sat_Fat':2.0,'Category':'salad','quantity':'1 bowl'}]}\n"
                )
            else:
                rules = (
                    "Rules:\n"
                    "1. Use only the provided item names, types, and quantities; do not invent or merge items.\n"
                    "2. For each item, estimate total nutrition for the visible serving size or stated weight.\n"
                    "3. Provide all required fields: Calories, Serving_Size, Protein, Carbs, Fiber, Sugar, Fat, Sat_Fat, Category.\n"
                    "4. Include optional Diet_Tags and Warnings when applicable.\n"
                    "5. Keep units consistent: grams for macros, kcal for energy.\n"
                    "6. Round numbers to one decimal where relevant.\n"
                    "7. If quantity is ambiguous, assume typical real-world weights (e.g., 1 egg ≈50 g).\n"
                    "8. When multiple pieces are listed, scale totals accordingly.\n"
                    "9. Return ONLY JSON matching the NutritionAnalysisResponse schema.\n"
                    "10. Do not include commentary, markdown, or extra keys.\n"
                )

                examples = (
                    "Examples:\n"
                    "Input: [{'name':'Knafeh','type':'main_dish','quantity':'1 piece ≈120 g'}]\n"
                    "Output:\n"
                    "{'combined':{'Calories':420,'Serving_Size':'1 piece ≈120 g','Protein':10.0,'Carbs':45.0,'Fiber':1.2,'Sugar':30.0,'Fat':20.0,'Sat_Fat':10.0,'Category':'dessert','Diet_Tags':['vegetarian'],'Warnings':['high sugar','high fat']},"
                    "'items':[{'name':'Knafeh','type':'main_dish','Calories':420,'Serving_Size':'1 piece ≈120 g','Protein':10.0,'Carbs':45.0,'Fiber':1.2,'Sugar':30.0,'Fat':20.0,'Sat_Fat':10.0,'Category':'dessert','Diet_Tags':['vegetarian'],'Warnings':['high sugar','high fat'],'quantity':'1 piece'}]}"
                )

            # 5. Prompt assembly
            prompt = (
                "Analyze these meal components and return ONLY valid JSON matching the NutritionAnalysisResponse schema.\n"
                + rules
                + examples
                + "\nItems:\n"
                + key_material
            )

            ts = datetime.utcnow().isoformat() + "Z"
            start = time.perf_counter()
            logger.info(
                "[OpenAI][START] ts=%s step=nutrition model=%s temp=%s max_tokens=%s items=%d hash=%s",
                ts, self.CHAT_MODEL, self.JSON_TEMPERATURE, self.JSON_MAX_TOKENS_NUTRITION, len(payload), cache_key
            )

            # 6. OpenAI call with enforced JSON schema validation and timeout
            try:
                resp = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.CHAT_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a meticulous nutrition scientist. "
                                    "Return ONLY valid JSON strictly conforming to the provided schema. "
                                    "No markdown, no comments."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                    "name": "NutritionAnalysisResponse",
                    "schema": schema_dict
                },  # enforce your NutritionAnalysisResponse structure
                        },
                        temperature=self.JSON_TEMPERATURE,
                        max_tokens=self.JSON_MAX_TOKENS_NUTRITION,
                    ),
                    timeout=45.0  # Increased timeout to 45 seconds for nutrition analysis
                )
            except asyncio.TimeoutError:
                logger.error("OpenAI nutrition analysis timed out after 45 seconds")
                raise Exception("Nutrition analysis timed out - this may be due to network issues or API overload")
            except Exception as e:
                logger.error("OpenAI nutrition analysis failed: %s", e)
                raise Exception(f"Nutrition analysis failed: {str(e)}")

            duration_ms = int((time.perf_counter() - start) * 1000)

            # 7. Defensive checks and parsing
            if not getattr(resp, "choices", None) or not resp.choices:
                raise ValueError("Empty response choices from nutrition model")

            content = getattr(resp.choices[0].message, "content", None)
            if not content:
                raise ValueError("Empty message content from nutrition model")

            raw = self._clean_ai_response(content)
            
            # Handle missing Category field in fast mode
            try:
                parsed = NutritionAnalysisResponse.model_validate_json(raw)
            except Exception as validation_error:
                if "Category" in str(validation_error) and self.FAST_MODE:
                    logger.warning("Adding missing Category fields for fast mode compatibility")
                    # Parse as dict and add missing categories
                    import json as json_module
                    data = json_module.loads(raw)
                    
                    # Add Category to combined if missing
                    if "combined" in data and "Category" not in data["combined"]:
                        data["combined"]["Category"] = "mixed"
                    
                    # Add Category to items if missing
                    if "items" in data:
                        for item in data["items"]:
                            if "Category" not in item:
                                # Guess category from type
                                item_type = item.get("type", "unknown")
                                category_map = {
                                    "main_dish": "entree",
                                    "side_dish": "side",
                                    "beverage": "drink",
                                    "condiment": "sauce",
                                    "unknown": "other"
                                }
                                item["Category"] = category_map.get(item_type, "other")
                    
                    # Try parsing again
                    raw = json_module.dumps(data)
                    parsed = NutritionAnalysisResponse.model_validate_json(raw)
                else:
                    raise validation_error
                    
            result = parsed.to_nutrition_analysis()
            self._nutrition_cache[cache_key] = result

            logger.info(
                "[OpenAI][END] step=nutrition dur_ms=%d hash=%s items=%d",
                duration_ms, cache_key, len(result.items if hasattr(result, "items") else []),
            )
            return result

        except Exception as e:
            logger.error("Error in _analyze_nutrition: %s", e, exc_info=True)
            
            # Return a basic fallback result in fast mode
            if self.FAST_MODE and "Category" in str(e):
                logger.warning("Returning fallback nutrition data due to validation error in fast mode")
                # Create a minimal valid response
                fallback_nutrition = NutritionInfo(
                    calories=300,
                    serving_size="1 serving",
                    protein_g=20.0,
                    carbs_g=30.0,
                    fiber_g=5.0,
                    sugar_g=10.0,
                    fat_g=12.0,
                    saturated_fat_g=3.0,
                    category="mixed",
                    diet_tags=[],
                    warnings=["Estimated values due to fast mode processing"]
                )
                
                fallback_item = FoodItemWithNutrition(
                    name="Mixed Meal",
                    type="main_dish",
                    nutrition=fallback_nutrition
                )
                
                return NutritionAnalysis(
                    combined=fallback_nutrition,
                    items=[fallback_item]
                )
            
            return None

    async def _get_recommendations(self, nutrition: Optional[Any], user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            class SimpleNutrition:
                def __init__(self, calories: float = 0, protein_g: float = 0, carbs_g: float = 0, fat_g: float = 0):
                    self.calories = calories
                    self.protein_g = protein_g
                    self.carbs_g = carbs_g
                    self.fat_g = fat_g
            
            if nutrition and hasattr(nutrition, "combined"):
                combined = nutrition.combined
            elif isinstance(nutrition, dict) and "nutrition_summary" in nutrition:
                ns = nutrition["nutrition_summary"]
                combined = SimpleNutrition(
                    calories=ns.get("calories", {}).get("value", 0),
                    protein_g=ns.get("macros", {}).get("protein", {}).get("value", 0),
                    carbs_g=ns.get("macros", {}).get("carbs", {}).get("value", 0),
                    fat_g=ns.get("macros", {}).get("fat", {}).get("value", 0),
                )
            else:
                combined = SimpleNutrition()

            meal_summary = (
                f"Calories {combined.calories} kcal, Protein {combined.protein_g}g, "
                f"Carbs {combined.carbs_g}g, Fat {combined.fat_g}g"
            )

            # 2. Versioned cache key for recommendations (coarse granularity by macro quartet)
            prompt_version = "rec_v4"  # bumped to v4 for enhanced diet-specific suggestions
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

            user_context = _get_user_context(user_preferences)

            # 4. Rules & JSON contract (intelligent suggestions)
            # 4. Rules & JSON contract (explicit health-aware guidance)
            contract = (
    "OUTPUT FORMAT (strict JSON): {\n"
    "  'recommendations': [str],\n"
    "  'health_score': int (0–100),\n"
    "  'meal_type': str ('breakfast'|'lunch'|'dinner'|'snack'|'unknown'),\n"
    "  'dietary_considerations': [str],\n"
    "  'meal_rating': int (0–10),\n"
    "  'suggestions': [str],\n"
    "  'improvements': [str],\n"
    "  'positive_aspects': [str]\n"
    "}"
)

            rules = (
    "EVALUATION RULES:\n"
    "1. Use only the provided macros (calories, protein, carbs, fat).\n"
    "2. health_score = overall nutritional quality (higher = better).\n"
    "HEALTH SCORE CALIBRATION:\n"
    "- 90-100 = excellent macro balance, nutrient dense, supports goal.\n"
    "- 75-89 = good nutrition, minor improvements possible.\n"
    "- 60-74 = fair but unbalanced.\n"
    "- 40-59 = poor nutritional quality.\n"
    "- <40 = very poor.\n"
    "CONDITION-ADJUSTED SCORING:\n"
    "- Score must reflect user health conditions.\n"
    "- Diabetes: high sugar/simple carbs → <60; fiber/protein-balanced → 80-95.\n"
    "- Hypertension: high sodium → <55; potassium-rich, low-sodium → 80-95.\n"
    "- Heart/Cholesterol: sat-fat-heavy → <60; omega-3/high-fiber → 80-95.\n"
    "- Kidney: excessive protein → <60; moderate protein, low sodium → 75-90.\n"
    "- Celiac/Gluten: gluten present → 0; gluten-free → 80-95.\n"
    "- Lactose intolerance: dairy → <60; lactose-free → 80-95.\n"
    "- IBS: high-FODMAP → <60; low-FODMAP → 75-90.\n"
    "- Multiple conditions: apply strictest applicable rule.\n"
    "3. meal_rating = taste/appeal (0-10).\n"
    "4. Reinforce good balance with 'positive_aspects'; avoid nitpicking healthy meals.\n"
    "5. Suggest improvements only when macros or goals conflict.\n"

    "6. HEALTH CONDITIONS HAVE HIGHEST PRIORITY. Always evaluate and filter suggestions through user conditions first.\n"
    "   - If a meal conflicts with any health condition, assign health_score <50.\n"
    "   - Never praise or label 'great' any meal that violates Keto, diabetic, or heart-healthy rules.\n"
    "   - Provide clear improvement suggestions to make it compliant.\n"

    "7. Respect goals:\n"
    "   - Cutting → reduce calories, preserve satiety.\n"
    "   - Bulking → increase calories, maintain high protein.\n"
    "   - Maintenance/Balanced → moderate everything.\n"
    "8. Respect diets strictly (Vegan, Keto, Mediterranean, etc.).\n"
    "9. If multiple conditions exist, combine restrictions conservatively (never conflict).\n"
    "10. Always fill every list with ≥1 element.\n"
    "11. Output only valid JSON, no markdown or explanations."
)


            examples = (
    "EXAMPLES:\n"
    "- Balanced (300 kcal, 25g protein): suggestions=['Good protein balance'], improvements=[], positive_aspects=['Healthy macros']\n"
    "- High-fat (800 kcal, 5g protein): improvements=['Add lean protein','Add vegetables for fiber']\n"
    "- Vegan: suggestions=['Excellent plant proteins'], improvements=['Add B12 source']\n"
    "- Keto: suggestions=['Strong fat-to-protein ratio'], improvements=['Add avocado for healthy fats']\n"
    "- Diabetic: suggestions=['Complex carbs aid glucose control'], improvements=['Pair with protein to slow absorption']\n"
    "- Hypertension: suggestions=['Low sodium helps heart health'], improvements=['Add potassium-rich foods like spinach']\n"
)

            prompt = (
    f"You are an AI nutrition coach. Be factual, evidence-based, and concise.\n"
    f"Analyze the following meal and user profile.\n\n"
    f"Meal Macros: {meal_summary}\n"
    f"{user_context}\n"
    f"{contract}\n"
    f"{rules}\n"
    f"{examples}\n"
    "Return ONLY valid JSON matching the schema above."
)


            ts = datetime.utcnow().isoformat() + "Z"
            start = time.perf_counter()
            logger.info(
                "[OpenAI][START] ts=%s step=recommendations model=%s temp=%s max_tokens=%s hash=%s summary='%s'",
                ts, self.CHAT_MODEL, self.JSON_TEMPERATURE, self.JSON_MAX_TOKENS, rec_hash, meal_summary
            )

            try:
                resp = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.CHAT_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a concise, evidence-based nutrition coach. Output ONLY JSON."},
                            {"role": "user", "content": prompt},
                        ],
                        response_format={"type": "json_object"},
                        temperature=self.JSON_TEMPERATURE,
                        max_tokens=self.JSON_MAX_TOKENS,
                    ),
                    timeout=45.0  # Increased timeout to 45 seconds for recommendations
                )
            except asyncio.TimeoutError:
                logger.error("OpenAI recommendations analysis timed out after 45 seconds")
                raise Exception("Recommendations analysis timed out - this may be due to network issues or API overload")
            except Exception as e:
                logger.error("OpenAI recommendations analysis failed: %s", e)
                raise Exception(f"Recommendations analysis failed: {str(e)}")

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
                arr_value = result.get(arr_key)
                if not isinstance(arr_value, list) or (arr_value is not None and len(arr_value) == 0):
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

    async def suggest_improvements(self, nutrition: Union[NutritionAnalysis, Dict[str, Any]], user_preferences: Optional[Dict[str, Any]] = None) -> List[str]:
        """Public helper to get quick improvement suggestions."""
        try:
            rec = await self._get_recommendations(nutrition, user_preferences)
            return rec.get("improvements", []) or rec.get("recommendations", [])
        except Exception as e:
            logger.error(f"Error in suggest_improvements: {e}")
            return ["No improvements available."]

    async def answer_question(self, question: str, meal_data: Dict[str, Any], user_preferences: Optional[Dict[str, Any]] = None):
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
                "7. IMPORTANT: Always consider user's health conditions when providing advice. For diabetes (focus on blood sugar management), hypertension (low sodium guidance), etc.\n"
                "8. Prioritize health condition considerations in your recommendations while maintaining encouraging tone.\n"
            )

            # Build user context from preferences
            user_context = ""
            if user_preferences:
                goal = user_preferences.get('goal', 'Balanced')
                diet = user_preferences.get('diet', 'None')
                health_conditions = user_preferences.get('health_conditions', [])
                calorie_target = user_preferences.get('calorie_target', 2500)
                
                user_context = f"\nUser Profile: Goal={goal}, Diet={diet}"
                if health_conditions:
                    user_context += f", Health Conditions={', '.join(health_conditions)}"
                user_context += f", Daily Target={calorie_target} kcal. Consider this context."

            user_msg = (
                f"Meal: {meal_name}. Nutrition summary: {nutrition_text}." +
                user_context + f"\n"
                f"Question: {question}"
            )

            # Type the messages properly for OpenAI
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]

            ts = datetime.utcnow().isoformat() + "Z"
            start = time.perf_counter()
            logger.info(
                "[OpenAI][START] ts=%s step=qa_stream model=%s meal=%s question_len=%d", ts, self.CHAT_MODEL, meal_name, len(question)
            )

            try:
                stream = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.CHAT_MODEL,
                        messages=messages,  # type: ignore
                        stream=True,
                        max_tokens=500,
                        temperature=0.5,
                    ),
                    timeout=45.0  # Increased timeout to 45 seconds for chat response
                )
            except asyncio.TimeoutError:
                logger.error("OpenAI streaming chat timed out after 45 seconds")
                raise Exception("Chat analysis timed out - this may be due to network issues or API overload")
            except Exception as e:
                logger.error("OpenAI streaming chat failed: %s", e)
                raise Exception(f"Chat analysis failed: {str(e)}")

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
