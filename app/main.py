from app.database import get_db, UserPreferences
import asyncio
import json
import logging
import os
import re
from typing import Optional, Tuple, List, Dict, Any
import hashlib

import nest_asyncio
import streamlit as st
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAIError
from PIL import Image
import io
import time

# Ensure project root is on sys.path so 'services' can be imported when run from different CWDs
import sys as _sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from services.analyzer import FoodAnalyzerService

# Better event loop management for debugging compatibility
try:
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply nest_asyncio only if needed
try:
    nest_asyncio.apply()
except Exception as e:
    logger.warning(f"Could not apply nest_asyncio: {e}")

load_dotenv()
CLIENT_AVAILABLE = False
client: Optional[AsyncOpenAI] = None

from app.config import VISION_MODEL, CHAT_MODEL

# Load icons from JSON file
def load_icons() -> Dict[str, str]:
    """Load icons from the icons.json file."""
    try:
        icons_path = os.path.join(os.path.dirname(__file__), 'icons.json')
        with open(icons_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('icons', {})
    except Exception as e:
        logger.warning(f"Could not load icons.json: {e}")
        return {}

def safe_run_async(coro):
    """
    Safely run async code in both normal execution and debugging environments.
    
    Args:
        coro: Async coroutine to run
        
    Returns:
        Result of the coroutine
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_running():
            # If loop is already running (e.g., in Jupyter/debugging), 
            # we need to use nest_asyncio
            import nest_asyncio
            nest_asyncio.apply()
            return current_loop.run_until_complete(coro)
        else:
            return current_loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop, create one
        try:
            return asyncio.run(coro)
        except RuntimeError:
            # Fallback: create new loop
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
    except Exception as e:
        logger.error(f"Error running async code: {e}")
        raise

# Global icons dictionary
ICONS = load_icons()

def get_icon(key: str, default: str = "🍴") -> str:
    """Get an icon for a given key, with fallback to default."""
    # Try exact match first
    if key.lower() in ICONS:
        return ICONS[key.lower()]
    
    # Try partial matches for compound terms
    for icon_key, icon_value in ICONS.items():
        if key.lower() in icon_key or icon_key in key.lower():
            return icon_value
    
    return default

async def verify_openai_access(client: AsyncOpenAI) -> Tuple[bool, Optional[str]]:
    try:
        response = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        if response.choices and len(response.choices) > 0:
            logger.info("Successfully verified OpenAI API access")
            return True, None
    except OpenAIError as e:
        error_msg = str(e)
        logger.error(f"OpenAI API error: {error_msg}")
        return False, error_msg
    except Exception as e:
        logger.error(f"Unexpected error testing OpenAI access: {e}")
        return False, str(e)
    
    return False, "No response received from OpenAI API"

async def process_image(image, analyzer: FoodAnalyzerService, user_preferences: Dict[str, Any]):
    try:
        analysis_result = await analyzer.analyze_meal(image, user_preferences)
        if not analysis_result or analysis_result.get("status") == "error":
            error_msg = analysis_result.get("error", "Unknown error") if analysis_result else "No result"
            logger.error(f"Analysis failed: {error_msg}")
            return {
                "dish_name": "Unknown Dish",
                "confidence": 0.0
            }, None
        meal_info = analysis_result.get("meal_info", {})
        components = analysis_result.get("components", [])
        dish_name = meal_info.get("name", "Unknown Dish")
        confidence = meal_info.get("confidence", 0.0)
        
        # Check for non-food items
        if dish_name in ["Non-Food Item Detected", "Unidentified Food"] or confidence == 0.0:
            logger.warning("Non-food item detected in uploaded image")
            return {
                "dish_name": "Non-Food Item",
                "confidence": 0.0,
                "is_non_food": True
            }, analysis_result
            
        if dish_name == "Unknown Dish" and components:
            dish_name = components[0].get("name", "Unknown Dish")
        logger.info(f"Successfully analyzed: {dish_name} (confidence: {confidence:.2f})")
        return {
            "dish_name": dish_name,
            "confidence": confidence,
            "is_non_food": False
        }, analysis_result
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {
            "dish_name": "Unknown Dish",
            "confidence": 0.0
        }, None

async def handle_chat(analyzer: FoodAnalyzerService, question: str, analysis_result: Dict[str, Any], user_preferences: Dict[str, Any]):
    import time
    from datetime import datetime
    
    _start = time.perf_counter()  # Initialize at the beginning
    try:
        _ts = datetime.utcnow().isoformat()
        logger.info(f"[Chat][START] ts={_ts} question_len={len(question)}")
        stream = await analyzer.answer_question(question, analysis_result, user_preferences)
        token_count = 0
        async for chunk in stream:
            token_count += 1
            if token_count == 1:
                logger.info(f"[Chat][FIRST_TOKEN] received after {int((time.perf_counter() - _start) * 1000)}ms")
            yield chunk
        _dur_ms = int((time.perf_counter() - _start) * 1000)
        logger.info(f"[Chat][END] dur_ms={_dur_ms} tokens={token_count}")
    except Exception as e:
        _dur_ms = int((time.perf_counter() - _start) * 1000)
        logger.error(f"[Chat][ERROR] dur_ms={_dur_ms} error={str(e)}")
        yield f"Sorry, I encountered an error: {str(e)}"

def generate_chat_suggestions(user_preferences: Dict[str, Any], analyzer: Optional[FoodAnalyzerService] = None) -> List[str]:
    """Generate AI-powered personalized chat suggestions based on user preferences"""
    
    # Fallback suggestions if AI generation fails or analyzer is not available
    fallback_suggestions = [
        "Is this a well-balanced meal for my goals?",
        "What are the main nutritional benefits of this meal?",
        "How can I improve this meal to better suit my needs?"
    ]
    
    # If no analyzer available (demo mode), return fallback
    if analyzer is None:
        return fallback_suggestions
    
    try:
        # Build user persona description
        goal = user_preferences.get('goal', 'Balanced')
        diet = user_preferences.get('diet', 'None')
        health_conditions = user_preferences.get('health_conditions', [])
        calorie_target = user_preferences.get('calorie_target', 2500)
        
        # Create persona context
        persona_parts = [f"Goal: {goal}"]
        if diet != 'None':
            persona_parts.append(f"Diet: {diet}")
        if health_conditions:
            persona_parts.append(f"Health Conditions: {', '.join(health_conditions)}")
        persona_parts.append(f"Daily Calorie Target: {calorie_target} kcal")
        
        persona_context = " | ".join(persona_parts)
        
        # Check session state cache first to avoid repeated API calls
        if 'chat_suggestions_cache' not in st.session_state:
            st.session_state.chat_suggestions_cache = {}
        
        cache_key = persona_context
        if cache_key in st.session_state.chat_suggestions_cache:
            logger.info("Using cached chat suggestions for user preferences")
            return st.session_state.chat_suggestions_cache[cache_key]
        
        # Generate suggestions using AI
        logger.info("Generating new AI-powered chat suggestions")
        suggestions = safe_run_async(_generate_ai_suggestions(analyzer, persona_context))
        
        # Validate and cache suggestions
        if suggestions and len(suggestions) >= 3:
            final_suggestions = suggestions[:3]
            st.session_state.chat_suggestions_cache[cache_key] = final_suggestions
            return final_suggestions
        else:
            return fallback_suggestions
            
    except Exception as e:
        logger.warning(f"Failed to generate AI suggestions: {e}")
        return fallback_suggestions

async def _generate_ai_suggestions(analyzer: FoodAnalyzerService, persona_context: str) -> List[str]:
    """Generate AI-powered chat suggestions using OpenAI"""
    try:
        logger.info(f"Generating AI suggestions for persona: {persona_context}")
        
        prompt = f"""You are a nutrition coach helping users ask relevant questions about their meal analysis.

User Profile: {persona_context}

Generate exactly 3 short, actionable questions (max 12 words each) that this user would want to ask about their meal analysis. 

Guidelines:
1. Questions should be relevant to their specific goals, diet, and health conditions
2. Make questions practical and actionable
3. Avoid generic questions - personalize for this user profile
4. Keep questions conversational and natural
5. Focus on areas where this user would need guidance

Examples of good personalized questions:
- For Keto diet: "Are the carbs in this meal too high for keto?"
- For Bulking goal: "Does this meal have enough protein for muscle growth?"
- For Diabetes: "Will this meal spike my blood sugar levels?"
- For Cutting: "How can I reduce calories without losing satisfaction?"

Return ONLY a JSON array of exactly 3 question strings, no other text:
["question 1", "question 2", "question 3"]"""

        start_time = time.time()
        response = await asyncio.wait_for(
            analyzer.client.chat.completions.create(
                model=analyzer.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful nutrition coach. Return ONLY valid JSON with exactly 3 personalized questions."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=200,
            ),
            timeout=30.0  # 30 second timeout for suggestion generation
        )
        
        duration = time.time() - start_time
        logger.info(f"AI suggestion generation completed in {duration:.2f}s")
        
        if not response.choices or not response.choices[0].message.content:
            raise ValueError("Empty response from OpenAI")
            
        content = response.choices[0].message.content.strip()
        logger.debug(f"Raw AI response: {content}")
        
        # Clean and parse the response
        if content.startswith('```'):
            content = content.strip('`').replace('json', '').strip()
        
        # Parse the JSON response
        import json
        try:
            # Try direct JSON parsing first
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                # Handle various response structures
                if 'questions' in parsed:
                    suggestions = parsed['questions']
                elif 'suggestions' in parsed:
                    suggestions = parsed['suggestions']
                elif any(key in parsed for key in ['question1', 'question2', 'question3']):
                    suggestions = [parsed.get(f'question{i}', '') for i in range(1, 4)]
                    suggestions = [s for s in suggestions if s]  # Remove empty strings
                else:
                    # If it's a dict but no recognized structure, take first 3 values that look like questions
                    suggestions = [v for v in parsed.values() if isinstance(v, str) and '?' in v][:3]
            elif isinstance(parsed, list):
                suggestions = parsed
            else:
                raise ValueError("Invalid JSON structure")
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}. Attempting text extraction from: {content[:100]}...")
            # Try to extract questions from text if JSON parsing fails
            suggestions = []
            
            # Try various text patterns
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                # Remove numbering and quotes
                clean_line = re.sub(r'^\d+\.?\s*', '', line)  # Remove "1. " prefix
                clean_line = clean_line.strip('\'"')  # Remove quotes
                
                if len(clean_line) > 10 and ('?' in clean_line or any(word in clean_line.lower() for word in ['how', 'what', 'is', 'does', 'can', 'should', 'will'])):
                    suggestions.append(clean_line)
            
            # If still no suggestions, try comma-separated format
            if not suggestions and ',' in content:
                parts = content.split(',')
                for part in parts:
                    clean_part = part.strip().strip('\'"')
                    if len(clean_part) > 10:
                        suggestions.append(clean_part)
        
        # Validate suggestions
        valid_suggestions = []
        for suggestion in suggestions:
            if isinstance(suggestion, str) and len(suggestion.strip()) > 5:
                # Clean up the suggestion
                clean_suggestion = suggestion.strip()
                # Remove any remaining quotes or brackets
                clean_suggestion = re.sub(r'^[\[\]"\']|[\[\]"\']$', '', clean_suggestion)
                # Ensure it ends with a question mark
                if not clean_suggestion.endswith('?'):
                    clean_suggestion += '?'
                # Capitalize first letter
                if clean_suggestion:
                    clean_suggestion = clean_suggestion[0].upper() + clean_suggestion[1:]
                valid_suggestions.append(clean_suggestion)
        
        if len(valid_suggestions) >= 3:
            logger.info(f"Generated {len(valid_suggestions)} valid AI suggestions: {valid_suggestions[:3]}")
            return valid_suggestions[:3]
        else:
            raise ValueError(f"Not enough valid suggestions generated. Got {len(valid_suggestions)}: {valid_suggestions}")
            
    except asyncio.TimeoutError:
        logger.error("AI suggestion generation timed out after 30 seconds")
        raise Exception("Suggestion generation timed out - this may be due to network issues or API overload")
    except Exception as e:
        logger.error(f"Error generating AI suggestions: {e}")
        raise

def load_styles():
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" crossorigin="anonymous" referrerpolicy="no-referrer" />
        """,
        unsafe_allow_html=True,
    )
    theme_js = """
    <script>
    (function(){
      try {
        const stored = window.localStorage.getItem('fitplate-theme');
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const theme = stored || (prefersDark ? 'dark' : 'light');
        document.documentElement.setAttribute('data-theme', theme);
      } catch (e) {}
    })();
    </script>
    """
    st.markdown(theme_js, unsafe_allow_html=True)
    try:
      with open(os.path.join(os.path.dirname(__file__), 'styles.css')) as f:
          st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
      st.warning(f"Could not load styles.css: {e}")


def theme_toggle():
    if 'theme' not in st.session_state:
        st.session_state.theme = None
    current = st.session_state.theme
    if not current:
        current = 'dark'
    colA, colB = st.columns([6,1])
    with colB:
        toggle = st.toggle("Dark Mode", value=(current=='dark'), label_visibility="collapsed")
    theme = 'dark' if toggle else 'light'
    st.session_state.theme = theme
    st.markdown(
        f"""
        <script>
          try {{
            document.documentElement.setAttribute('data-theme', '{theme}');
            window.localStorage.setItem('fitplate-theme', '{theme}');
          }} catch(e) {{}}
        </script>
        """,
        unsafe_allow_html=True,
    )

def init_preferences():
    if 'prefs' not in st.session_state:
        user_id = None
        if 'user_id' in st.session_state:
            user_id = st.session_state.user_id
        else:
            try:
                from app.auth import get_current_user_id
                user_id = get_current_user_id()
            except Exception:
                user_id = None
        st.session_state.prefs = fetch_user_preferences(user_id)
def fetch_user_preferences(user_id):
    """Fetch user preferences from DB, fallback to defaults if not found."""
    db = get_db()
    prefs = {'goal': 'Balanced', 'diet': 'None', 'health_conditions': [], 'calorie_target': 2500}
    if user_id:
        try:
            pref_obj = db.query(UserPreferences).filter_by(user_id=user_id).first()
            if pref_obj:
                prefs = {
                    'goal': pref_obj.goal,
                    'diet': pref_obj.diet_type,
                    'health_conditions': pref_obj.health_conditions or [],
                    'calorie_target': pref_obj.calorie_target
                }
        except Exception as e:
            pass
    return prefs
    st.markdown(
        """
        <script>
        try {
          const raw = window.localStorage.getItem('fitplate-prefs');
          if (raw) {
            const prefs = JSON.parse(raw);
            const pySet = (k, v) => {
              const el = document.createElement('div');
              el.setAttribute('data-key', k);
              el.setAttribute('data-value', JSON.stringify(v));
              el.id = 'prefs-' + k;
              document.body.appendChild(el);
            };
            for (const [k, v] of Object.entries(prefs)) pySet(k, v);
          }
        } catch(e) {}
        </script>
        """,
        unsafe_allow_html=True,
    )

def preferences_sidebar():
    with st.sidebar:
        st.markdown("### 🎯 Preferences")
        goal = st.selectbox("Goal", ["Balanced", "Bulking", "Cutting", "Maintenance"], index=["Balanced","Bulking","Cutting","Maintenance"].index(st.session_state.prefs.get('goal','Balanced')))
        diet = st.selectbox("Diet", ["None", "Keto", "Vegan", "Vegetarian", "Mediterranean"], index=["None","Keto","Vegan","Vegetarian","Mediterranean"].index(st.session_state.prefs.get('diet','None')))
        health_conditions = st.multiselect("Health Conditions", ["Diabetes", "Heart disease", "High blood pressure", "High cholesterol", "Celiac Disease", "Lactose Intolerance"], default=st.session_state.prefs.get('health_conditions', []))
        cal = st.text_input("Daily calorie target", value=str(st.session_state.prefs.get('calorie_target') or '2500'))
        cal_val = None
        try:
            cal_val = int(cal) if cal.strip() else None
        except Exception:
            pass
        # Update session state
        st.session_state.prefs.update({'goal': goal, 'diet': diet, 'health_conditions': health_conditions, 'calorie_target': cal_val})
        # Sync to DB
        user_id = None
        if 'user_id' in st.session_state:
            user_id = st.session_state.user_id
        else:
            try:
                from app.auth import get_current_user_id
                user_id = get_current_user_id()
            except Exception:
                user_id = None
        if user_id:
            save_user_preferences(user_id, st.session_state.prefs)
        st.markdown(
            f"""
            <script>
            try {{
              const prefs = {json.dumps({'goal': goal, 'diet': diet, 'health_conditions': health_conditions, 'calorie_target': cal_val})};
              window.localStorage.setItem('fitplate-prefs', JSON.stringify(prefs));
            }} catch(e) {{}}
            </script>
            """,
            unsafe_allow_html=True,
        )
def save_user_preferences(user_id, prefs):
    """Save user preferences to DB."""
    db = get_db()
    try:
        pref_obj = db.query(UserPreferences).filter_by(user_id=user_id).first()
        if not pref_obj:
            pref_obj = UserPreferences(user_id=user_id)
            db.add(pref_obj)
        pref_obj.goal = prefs.get('goal', 'Balanced')
        pref_obj.diet_type = prefs.get('diet', 'None')
        pref_obj.health_conditions = prefs.get('health_conditions', [])
        pref_obj.calorie_target = prefs.get('calorie_target', 2500)
        db.commit()
    except Exception as e:
        db.rollback()

def _load_demo_from_json(path: str) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        dish = data.get('dish') or {"dish_name": "Sample Meal", "confidence": 0.85}
        nutrition = data.get('nutrition') or {}
        suggestions = data.get('suggestions') or []
        return dish, nutrition, suggestions
    except Exception as e:
        logger.error(f"Failed to load demo JSON from {path}: {e}")
        # Fallback minimal shape to keep UI working
        fallback = {
            "meal_info": {"serving_size": "1 serving"},
            "nutrition_summary": {
                "calories": {"value": 400, "daily_value": 20},
                "macros": {
                    "protein": {"value": 25, "daily_value": 50},
                    "carbs": {"value": 45, "daily_value": 15},
                    "fat": {"value": 12, "daily_value": 15}
                },
                "additional": {
                    "fiber": {"value": 6, "daily_value": 24},
                    "sugar": {"value": 12, "daily_value": 0},
                    "saturated_fat": {"value": 3, "daily_value": 15}
                },
                "warnings": ["Sodium may be high"],
                "diet_tags": ["balanced"]
            },
            "components": []
        }
        return {"dish_name": "Demo Meal", "confidence": 0.9}, fallback, ["Add a side of veggies for fiber."]


def _adjust_nutrition_for_portion(nutrition: Dict[str, Any], portion_consumed: float) -> Dict[str, Any]:
    """Adjust nutrition values based on portion consumed"""
    if portion_consumed == 1.0:
        return nutrition

    adjusted = nutrition.copy()

    # Adjust nutrition summary
    if "nutrition_summary" in adjusted:
        summary = adjusted["nutrition_summary"]

        # Adjust macros
        if "macros" in summary:
            for macro in summary["macros"]:
                if "value" in summary["macros"][macro]:
                    summary["macros"][macro]["value"] *= portion_consumed
                if "daily_value" in summary["macros"][macro]:
                    summary["macros"][macro]["daily_value"] *= portion_consumed

        # Adjust additional nutrients
        if "additional" in summary:
            for nutrient in summary["additional"]:
                if "value" in summary["additional"][nutrient]:
                    summary["additional"][nutrient]["value"] *= portion_consumed
                if "daily_value" in summary["additional"][nutrient]:
                    summary["additional"][nutrient]["daily_value"] *= portion_consumed

    # Calculate original total calories from components
    original_total_calories = 0
    if "components" in adjusted:
        for component in adjusted["components"]:
            if "nutrition" in component:
                original_total_calories += component["nutrition"].get("calories", 0)
        # Now adjust each component's nutrition by portion
        for component in adjusted["components"]:
            if "nutrition" in component:
                for nutrient, value in component["nutrition"].items():
                    if isinstance(value, (int, float)):
                        component["nutrition"][nutrient] = value * portion_consumed

    # Set nutrition_summary calories to total calories for portion consumed
    if "nutrition_summary" in adjusted and "calories" in adjusted["nutrition_summary"]:
        adjusted["nutrition_summary"]["calories"]["value"] = original_total_calories * portion_consumed
        # Optionally adjust daily_value as well (if desired)
        # adjusted["nutrition_summary"]["calories"]["daily_value"] *= portion_consumed

    return adjusted


def main():
    st.set_page_config(page_title="🍽️ Smart Dish Analyzer", page_icon="🍽️", layout="centered")
    
    # Import authentication functions
    from app.auth import is_authenticated, get_current_user_id, get_current_username
    from app.meal_tracking import MealTrackingService
    from app.database import init_database
    
    # Initialize database
    try:
        init_database()
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        logger.error(f"Database initialization error: {e}")
        return
    
    # Check authentication
    if not is_authenticated():
        st.warning("Please login to use FitPlate AI")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔐 Go to Login", type="primary", use_container_width=True):
                st.switch_page("pages/1_Login.py")
        return
    
    load_styles()
    init_preferences()
    preferences_sidebar()
    st.markdown('<div class="topbar"><div class="title">🍽️ FitPlate AI</div><div class="theme-toggle">', unsafe_allow_html=True)
    theme_toggle()
    st.markdown('</div></div>', unsafe_allow_html=True)
    # Demo mode toggle and JSON path
    demo_default = bool(os.getenv('UX_DEMO') or os.getenv('UX_DATA_JSON'))
    with st.sidebar:
        # Hide login if already authenticated
        from app.auth import is_authenticated
        if not is_authenticated():
            st.markdown("### 🔐 Login")
            if st.button("Go to Login", use_container_width=True):
                st.switch_page("pages/1_Login.py")
        st.markdown("### 🧪 Demo Mode")
        demo_mode = st.toggle("Run without API (JSON)", value=demo_default)
        json_path = st.text_input("Demo JSON path", value=os.getenv('UX_DATA_JSON', 'data/demo_meal.json')) if demo_mode else None
        st.markdown("### ⚡ Speed Settings")
        speed_mode = st.toggle("Fast Mode (Faster analysis, less detail)", value=False)
        if speed_mode:
            st.info("🚀 Fast mode: Reduced tokens, limited items, optimized prompts")

    # Initialize OpenAI client and analyzer only if not in demo mode
    analyzer: Optional[FoodAnalyzerService] = None
    if not demo_mode:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_key")
        if not api_key:
            st.error("No OpenAI API key found. Enable Demo Mode or set OPENAI_API_KEY in your .env file.")
            return
        try:
            global client, CLIENT_AVAILABLE
            client = AsyncOpenAI(api_key=api_key)
            CLIENT_AVAILABLE = True
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {e}")
            return

        access_ok, error_msg = safe_run_async(verify_openai_access(client))
        if not access_ok:
            st.error(f"⚠️ OpenAI API Error: {error_msg}")
            st.info("Turn on Demo Mode in the sidebar to run without API access.")
            return
        analyzer = FoodAnalyzerService(api_key=api_key)
        # Apply speed mode settings
        if 'speed_mode' in locals() and speed_mode:
            analyzer.enable_ultra_fast_mode()
    st.write("Upload a meal photo or use your camera to analyze your food and chat about your goals.")
    tab_upload, tab_camera = st.tabs(["📂 Upload Photo", "📷 Take Photo"]) 
    image = None
    # Initialize tracking for image sources
    if 'last_upload_ts' not in st.session_state:
        st.session_state.last_upload_ts = 0.0
    if 'last_camera_ts' not in st.session_state:
        st.session_state.last_camera_ts = 0.0
    with tab_upload:
        uploaded_file = st.file_uploader("Upload Food Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file is not None:
            up_bytes = uploaded_file.getvalue()
            up_hash = __import__('hashlib').md5(up_bytes).hexdigest()
            if st.session_state.get('last_upload_hash') != up_hash:
                st.session_state.last_upload_ts = time.time()
                st.session_state.last_upload_hash = up_hash
                st.session_state.upload_bytes = up_bytes
    with tab_camera:
        camera_input = st.camera_input("Take a photo")
        if camera_input is not None:
            cam_bytes = camera_input.getvalue()
            cam_hash = __import__('hashlib').md5(cam_bytes).hexdigest()
            if st.session_state.get('last_camera_hash') != cam_hash:
                st.session_state.last_camera_ts = time.time()
                st.session_state.last_camera_hash = cam_hash
                st.session_state.camera_bytes = cam_bytes

    # Decide which source to use based on latest timestamp
    chosen_source = None
    if st.session_state.last_upload_ts or st.session_state.last_camera_ts:
        chosen_source = 'upload' if st.session_state.last_upload_ts >= st.session_state.last_camera_ts else 'camera'

    selected_bytes = None
    if chosen_source == 'upload' and st.session_state.get('upload_bytes'):
        selected_bytes = st.session_state.upload_bytes
    elif chosen_source == 'camera' and st.session_state.get('camera_bytes'):
        selected_bytes = st.session_state.camera_bytes

    if selected_bytes:
        image = Image.open(io.BytesIO(selected_bytes))
        # Use session state to cache analysis results and avoid re-analyzing on chat interactions
        # Create a unique key for this image + user preferences
        import hashlib
        image_hash = hashlib.md5(selected_bytes).hexdigest()
        
        # Create preferences hash to invalidate suggestions when preferences change
        # Cache version: increment when backend analysis logic changes
        CACHE_VERSION = "v5"  # bumped for summation fix
        prefs_str = json.dumps(st.session_state.prefs, sort_keys=True)
        prefs_hash = hashlib.md5(prefs_str.encode()).hexdigest()
        cache_key = f"{CACHE_VERSION}_{image_hash}_{prefs_hash}"
        
        # Check if we already have results for this image + preferences combination
        if 'analysis_cache' not in st.session_state:
            st.session_state.analysis_cache = {}
        
        # Initialize variables
        dish = None
        nutrition = None
        ai_insights = None
        suggestions = []

        if cache_key in st.session_state.analysis_cache:
            cached_data = st.session_state.analysis_cache[cache_key]
            dish = cached_data['dish']
            nutrition = cached_data['nutrition']
            ai_insights = cached_data.get('ai_insights')
            suggestions = cached_data.get('suggestions', [])
        elif image_hash in [k.split('_')[0] for k in st.session_state.analysis_cache.keys()]:
            for key, cached_data in st.session_state.analysis_cache.items():
                if key.startswith(image_hash):
                    dish = cached_data['dish']
                    nutrition = cached_data['nutrition']
                    ai_insights = cached_data.get('ai_insights')
                    suggestions = cached_data.get('suggestions', [])
                    break
            # Do NOT call APIs again; just cache with new key if needed
            if dish is not None and nutrition is not None:
                st.session_state.analysis_cache[cache_key] = {
                    'dish': dish,
                    'nutrition': nutrition,
                    'ai_insights': ai_insights,
                    'suggestions': suggestions
                }
        else:
            loading_placeholder = st.empty()
            with loading_placeholder.container():
                st.markdown('<div class="progress-line"><div class="bar"></div></div>', unsafe_allow_html=True)
                st.markdown('<div class="ai-analyzing mt-2"><i class="fa-solid fa-wand-magic-sparkles"></i> Analyzing <span class="ai-dots"><span></span><span></span><span></span></span></div>', unsafe_allow_html=True)
                st.markdown('<div class="card"><div class="skel-grid"><div class="skel-line shimmer" style="height:24px"></div><div class="skel-line shimmer" style="height:24px"></div><div class="skel-line shimmer" style="height:24px"></div></div></div>', unsafe_allow_html=True)
            if demo_mode:
                with st.spinner("Loading demo analysis..."):
                    if json_path is not None:
                        dish, nutrition, suggestions = _load_demo_from_json(json_path)
                        ai_insights = None
                    else:
                        st.error("Demo JSON path not provided")
                        return
            else:
                with st.spinner("Analyzing your dish..."):
                    if analyzer is not None:
                        dish, nutrition = safe_run_async(process_image(image, analyzer, st.session_state.prefs))
                    else:
                        st.error("Analyzer not initialized")
                        return
                with st.spinner("Getting full AI insights..."):
                    if analyzer is not None:
                        full_recommendations = safe_run_async(analyzer._get_recommendations(nutrition, st.session_state.prefs))
                        ai_insights = full_recommendations
                        suggestions = full_recommendations.get('suggestions', [])
                        improvements = full_recommendations.get('improvements', [])
                        suggestions = improvements + suggestions
                    else:
                        ai_insights = None
                        suggestions = []
            loading_placeholder.empty()
            st.session_state.analysis_cache[cache_key] = {
                'dish': dish,
                'nutrition': nutrition,
                'ai_insights': ai_insights,
                'suggestions': suggestions
            }
        
        # Check if this is a non-food item
        if dish and (dish.get('is_non_food', False) or dish['dish_name'] == "Non-Food Item"):
            st.warning("🚫 **Non-Food Item Detected**")
            st.markdown("""
            It looks like the uploaded image doesn't contain food items. 
            
            **Please try uploading an image that contains:**
            - A meal or dish
            - Individual food items
            - Snacks or beverages
            
            For best results, ensure the food is clearly visible and well-lit.
            """)
            return
        
        # Always render the UI (using cached or fresh data)
        if not nutrition or not nutrition.get("nutrition_summary"):
            st.error("⚠️ Could not analyze nutrition information")
            return
        
        # Ensure we have valid dish data
        if not dish:
            st.error("⚠️ Could not analyze dish information")
            return
        
        # Merge ai_insights into nutrition if available
        if ai_insights:
            nutrition['ai_insights'] = ai_insights

        # Prepare common values for the left info column
        nutrition_summary = nutrition["nutrition_summary"]
        meal_info = nutrition.get("meal_info", {})
        confidence = dish['confidence'] * 100
        confidence_emoji = get_icon("confidence_high") if confidence >= 90 else get_icon("confidence_medium") if confidence >= 70 else get_icon("confidence_low")
        diet_tags = nutrition_summary.get("diet_tags", [])

        # Show left column (meal info) and right column (image)
        with st.container():
            st.markdown('<div class="card image-preview">', unsafe_allow_html=True)
            # Move the information to the left, image on the right
            col_info, col_img = st.columns([2, 3])
            with col_info:
                # Use only unified classes from styles.css

                # Meal card - total calories is sum of all components
                components_list = nutrition.get("components", [])
                total_calories = sum(
                    c.get("nutrition", {}).get("calories", 0)
                    for c in components_list
                )

                cal_dv = nutrition_summary["calories"]["daily_value"]
                health_score = None
                heart_icon = ""
                chip_color = ""
                if nutrition and 'ai_insights' in nutrition and 'health_score' in nutrition['ai_insights']:
                    health_score = nutrition['ai_insights']['health_score']
                if health_score is not None:
                    # Determine color and icon based on score
                    try:
                        score_val = float(health_score)
                    except Exception:
                        score_val = None
                    if score_val is not None:
                        if score_val >= 80:
                            chip_color = "background: linear-gradient(135deg, #b9f6ca 0%, #43ea7a 100%) !important; border-color: #43ea7a !important; color: #1B5E20 !important;"
                            heart_icon = "💚"
                        elif score_val >= 50:
                            chip_color = "background: linear-gradient(135deg, #fff59d 0%, #ffe082 100%) !important; border-color: #ffe082 !important; color: #795548 !important;"
                            heart_icon = "💛"
                        else:
                            chip_color = "background: linear-gradient(135deg, #ff8a80 0%, #ff5252 100%) !important; border-color: #ff5252 !important; color: #fff !important;"
                            heart_icon = "❤️"
                    else:
                        chip_color = ""
                        heart_icon = "💚"
                    score_html = f"<span style='{chip_color}'><span class='ico'>{heart_icon}</span> Health Score: {health_score}</span>"
                title_html = f"<div class='meal-title-row'><div class='meal-title'>{heart_icon} {dish['dish_name']}</div></div>"

                style_attr = f" style='{chip_color}'" if chip_color else ""
                meal_card_html = f"""
                    <div class='ui-card'{style_attr}>
                        {title_html}
                        <div class='stat-value'>{total_calories} <span>kcal</span></div>
                        <div class='stat-meter'><div class='fill' style='width:{cal_dv}%'></div></div>
                        <div class='stat-sub'>{cal_dv}% Daily Value</div>
                    </div>
                """
                st.markdown(meal_card_html, unsafe_allow_html=True)

                # Icon helpers using centralized icon system
                def _warn_icon(txt: str) -> str:
                    """Get warning icon based on text content using icons.json"""
                    t = txt.lower()
                    # Try to find the most specific match first
                    for keyword in ["saturated fat", "trans fat", "high sodium", "added sugar", "high calories", 
                                   "sodium", "salt", "sugar", "fat", "calorie", "fiber", "protein", "carb"]:
                        if keyword in t:
                            return get_icon(keyword, "⚠️")
                    return get_icon("allergen", "⚠️")

                def _sugg_icon(txt: str) -> str:
                    """Get suggestion icon based on text content using icons.json"""
                    t = txt.lower()
                    # Try to find the most specific match first
                    for keyword in ["add vegetables", "add fruits", "steamed vegetables", "grilled chicken", 
                                   "salmon steak", "avocado toast", "whole grain", "olive oil", "healthy fat",
                                   "vegetable", "veggie", "broccoli", "salad", "fruit", "protein", "chicken", 
                                   "fiber", "quinoa", "avocado", "water", "hydrate", "reduce calories"]:
                        if keyword in t:
                            return get_icon(keyword, "💡")
                    return get_icon("balanced", "💡")

                # Show AI Suggestions, Diet Tags, Health Score, and Warnings below the Dish info
                chips_html = ""
                # Suggestions chips
                if suggestions:
                    # Render all suggestions in a single card
                    sugg_chips = [
                        f"<span ><span class='ico'>{_sugg_icon(s)}</span>{s}</span>"
                        for s in suggestions
                    ]
                    st.markdown(
                        f"<div class='ui-card suggestions-card'><div class='ui-card-title'>AI Suggestions</div><div class='chips'>{''.join(sugg_chips)}</div></div>",
                        unsafe_allow_html=True,
                    )

                # Diet tags chips
                if diet_tags:
                    tag_chips = [
                        f"<span class='chip chip-tag'><span class='ico'>{get_icon(tag, get_icon('diet'))}</span>{tag.title()}</span>"
                        for tag in diet_tags
                    ]
                    chips_html += ''.join(tag_chips)

                # Health score chip with dynamic color
                

                # Render all chips together
                if chips_html:
                    st.markdown(f"<div class='chips'>{chips_html}</div>", unsafe_allow_html=True)

                # Warnings chips
                warn_list = nutrition_summary.get("warnings", [])
                if warn_list:
                    warn_chips = [
                        f"<span class='chip chip-warn'><span class='ico'>{_warn_icon(w)}</span>{w}</span>"
                        for w in warn_list
                    ]
                    st.markdown(
                        f"<div class='chips'>{''.join(warn_chips)}</div>",
                        unsafe_allow_html=True,
                    )

            with col_img:
                st.image(image, caption="Your Dish", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Row: Left = Additional Nutrients, Right = Macronutrients (aligned with image width and equal height)
        with st.container():
            # Use only unified classes from styles.css
            col_left, col_right = st.columns([2, 3])
            with col_left:
                additional = nutrition_summary.get("additional", {})
                fiber = additional.get("fiber", {})
                sugar = additional.get("sugar", {})
                sat = additional.get("saturated_fat", {})
                add_chips = [
                    f"<span class='chip'><span class='ico'>{get_icon('fiber')}</span>Fiber: {fiber.get('value', 0):.1f}g • {fiber.get('daily_value', 0)}% DV</span>",
                    f"<span class='chip'><span class='ico'>{get_icon('sugar')}</span>Sugar: {sugar.get('value', 0):.1f}g</span>",
                    f"<span class='chip'><span class='ico'>{get_icon('saturated fat')}</span>Saturated Fat: {sat.get('value', 0):.1f}g • {sat.get('daily_value', 0)}% DV</span>",
                ]
                st.markdown(
                    f"<div class='ui-card eq'><div class='ui-card-title'>{get_icon('additional_nutrients')} Additional Nutrients</div><div class='chips'>{''.join(add_chips)}</div></div>",
                    unsafe_allow_html=True,
                )
            with col_right:
                macros = nutrition_summary["macros"]
                macro_html = f"""
                    <div class='ui-card eq'>
                        <div class='ui-card-title'>{get_icon('nutrition')} Macronutrients</div>
                        <div class='chips'>
                          <span class='chip'><span class='ico'>{get_icon('protein')}</span>Protein: {macros['protein']['value']:.1f}g • {macros['protein']['daily_value']}% DV</span>
                          <span class='chip'><span class='ico'>{get_icon('carb')}</span>Carbs: {macros['carbs']['value']:.1f}g • {macros['carbs']['daily_value']}% DV</span>
                          <span class='chip'><span class='ico'>{get_icon('healthy fat')}</span>Fat: {macros['fat']['value']:.1f}g • {macros['fat']['daily_value']}% DV</span>
                        </div>
                    </div>
                """
                st.markdown(macro_html, unsafe_allow_html=True)

        
        # Macronutrients are shown next to Additional Nutrients above to align with the image column
        components = nutrition.get("components", [])
        if components:
            with st.expander(f"{get_icon('meal_components')} Meal Components", expanded=False):
                for item in components:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    # Use centralized icon system for component types
                    icon = get_icon(item['type'], get_icon('unknown'))
                    st.markdown(f"### {icon} {item['name']}")
                    st.caption(f"Type: {item['type'].replace('_', ' ').title()} | {item['serving']}")
                    c1, c2, c3 = st.columns(3)
                    n = item['nutrition']
                    c1.metric("Calories", f"{n['calories']} kcal")
                    c2.metric("Protein", f"{n['protein']:.1f}g")
                    c3.metric("Carbs", f"{n['carbs']:.1f}g")
                    with st.expander("More Details"):
                        d1, d2 = st.columns(2)
                        d1.metric("Fat", f"{n['fat']:.1f}g"); d1.metric("Fiber", f"{n['fiber']:.1f}g")
                        d2.metric("Sugar", f"{n['sugar']:.1f}g"); d2.metric("Saturated Fat", f"{n['saturated_fat']:.1f}g")
                        if item.get('diet_tags'):
                            tags_html = " ".join([f"<span class='diet-tag'>{tag}</span>" for tag in item['diet_tags']])
                            st.markdown(f"<div>**Diet Tags:** {tags_html}</div>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        # Warnings and AI Suggestions are shown near the image above.
        
        # Meal Confirmation Section - Improved UI
        st.markdown('<hr/>', unsafe_allow_html=True)

        user_id = get_current_user_id()
        if not user_id:
            st.error("User session invalid. Please login again.")
            return

        meal_key = f"meal_{cache_key}"
        meal_confirmed = st.session_state.get(f"{meal_key}_confirmed", False)

        with st.container():

            if not meal_confirmed:
                colA, colB = st.columns([2,1])
                with colA:
                    portion_options = {
                        "100%": 1.0,
                        "75%": 0.75,
                        "50%": 0.5,
                        "25%": 0.25,

                    }
                    portion_label = st.radio(
                        "Select portion eaten",
                        options=list(portion_options.keys()),
                        index=0,
                        key=f"{meal_key}_portion_radio"
                    )
                    portion_consumed = portion_options[portion_label]
                with colB:
                    confirm_meal = st.button(
                        f"{get_icon('confirm', '✅')} Confirm Meal",
                        type="primary",
                        key=f"{meal_key}_confirm",
                        help="Save this meal to your nutrition tracking"
                    )
                

                if confirm_meal:
                    try:
                        import hashlib
                        meal_info = nutrition.get("meal_info", {})
                        meal_name = meal_info.get("name", dish.get("dish_name", "Unknown Meal"))
                        components = nutrition.get("components", [])
                        confidence = dish.get("confidence", 0.0)
                        image_hash = hashlib.md5(selected_bytes).hexdigest()
                        adjusted_nutrition = _adjust_nutrition_for_portion(nutrition, portion_consumed)
                        health_score = None
                        if nutrition and 'ai_insights' in nutrition and 'health_score' in nutrition['ai_insights']:
                            health_score = nutrition['ai_insights']['health_score']
                        meal = MealTrackingService.save_meal_analysis(
                            user_id=user_id,
                            meal_name=meal_name,
                            nutrition_data=adjusted_nutrition,
                            components=components,
                            image_hash=image_hash,
                            confidence_score=confidence,
                            ai_suggestions=suggestions,
                            health_score=health_score
                        )
                        if meal:
                            success = MealTrackingService.confirm_meal(
                                meal_id=meal.id,  # type: ignore
                                user_id=user_id
                            )
                            if success:
                                st.session_state[f"{meal_key}_confirmed"] = True
                                st.session_state[f"{meal_key}_meal_id"] = meal.id
                                st.success(f"{get_icon('success', '🎉')} Meal saved to your tracking system!")
                                st.rerun()
                            else:
                                st.error("Error confirming meal")
                        else:
                            st.error("Error saving meal analysis")
                    except Exception as e:
                        st.error(f"Error saving meal: {str(e)}")
                        logger.error(f"Error confirming meal: {e}")
            else:
                meal_id = st.session_state.get(f"{meal_key}_meal_id")
                st.success(f"{get_icon('success', '✅')} This meal has been confirmed and added to your tracking!")
            st.markdown("</div>", unsafe_allow_html=True)
 
        # Chat section - available only when API is enabled
        if nutrition and not demo_mode:
            st.markdown('<hr/>', unsafe_allow_html=True)
            st.markdown('<div class="subsection-header">Ask about your meal</div>', unsafe_allow_html=True)
            
            # Generate personalized chat suggestions with loading indicator
            persona_parts = [f"Goal: {st.session_state.prefs.get('goal', 'Balanced')}"]
            if st.session_state.prefs.get('diet', 'None') != 'None':
                persona_parts.append(f"Diet: {st.session_state.prefs.get('diet')}")
            if st.session_state.prefs.get('health_conditions', []):
                persona_parts.append(f"Health Conditions: {', '.join(st.session_state.prefs.get('health_conditions', []))}")
            persona_context = " | ".join(persona_parts)
            
            # Check if we need to generate new suggestions
            cache_key = persona_context
            suggestions_cached = (
                'chat_suggestions_cache' in st.session_state and 
                cache_key in st.session_state.chat_suggestions_cache
            )
            
            if not suggestions_cached:
                with st.spinner("🤖 Generating personalized questions..."):
                    chat_suggestions = generate_chat_suggestions(st.session_state.prefs, analyzer)
            else:
                chat_suggestions = generate_chat_suggestions(st.session_state.prefs, analyzer)
            
            # Render all chat suggestions in a single card, buttons grouped together
            preset = None
            if chat_suggestions:
                # Render each chat suggestion in a separate card/button
                cols = st.columns(len(chat_suggestions))
                preset = None
                for i, s in enumerate(chat_suggestions):
                    with cols[i]:
                        if st.button(s, key=f"suggest_{i}"):
                            preset = s

            # Always show chat input, but use preset if button was clicked
            chat_input = st.chat_input("💬 Ask about your meal")
            user_question = preset if preset else chat_input

            if user_question:
                with st.chat_message("user", avatar="🤔"):
                    st.write(user_question)
                with st.chat_message("assistant", avatar="🍽️"):
                    async def stream_answer():
                        if analyzer is not None:
                            async for token in handle_chat(analyzer, user_question, nutrition, st.session_state.prefs):
                                yield token
                        else:
                            yield "Sorry, the analyzer is not available in demo mode."
                    st.write_stream(stream_answer())
        elif nutrition and demo_mode:
            st.markdown('<hr/>', unsafe_allow_html=True)
            st.info("Chat is disabled in Demo Mode. Disable Demo Mode in the sidebar to ask questions.")

if __name__ == "__main__":
    main()
