import asyncio
import json
import logging
import os
import re
from typing import Optional, Tuple, List, Dict, Any

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

def render_meal_analysis(analysis_data):
    if analysis_data["status"] == "error":
        st.error(f"Analysis failed: {analysis_data['error']}")
        return
    with st.container():
        st.title("🍽️ Meal Analysis")
        meal = analysis_data["meal_info"]
        st.success(f"📸 Analyzed: {meal['name']} ({meal['confidence']:.1f}% confidence)")
    with st.container():

        col1, col2 = st.columns([2, 3])
        with col1:
            nutrition = analysis_data["nutrition_summary"]
            st.metric(
                "Total Calories",
                f"{nutrition['calories']['value']} kcal",
                f"{nutrition['calories']['daily_value']}% DV"
            )
            st.caption(f"Serving: {meal['serving_size']}")
        with col2:
            if nutrition["diet_tags"]:
                tags_html = " ".join([
                    f'<span class="nutrient-tag">{tag}</span>'
                    for tag in nutrition["diet_tags"]
                ])
                st.markdown(f"<div style='margin-top: 0.5rem'>{tags_html}</div>", unsafe_allow_html=True)
    insights = analysis_data["ai_insights"]
    with st.container():
        st.markdown("### 🤖 AI Insights")
        score_color = "green" if insights["health_score"] >= 80 else "orange" if insights["health_score"] >= 60 else "red"
        st.markdown(f"**Health Score:** <span style='color: {score_color}'>{insights['health_score']}/100</span>", unsafe_allow_html=True)
        suggestions = analysis_data.get("suggestions") or insights.get("suggestions", [])
        for suggestion in suggestions:
            st.markdown(suggestion)
        if insights["dietary_considerations"]:
            with st.expander("📋 Dietary Considerations"):
                for consideration in insights["dietary_considerations"]:
                    st.markdown(f"• {consideration}")
    with st.container():
        st.markdown("### 📊 Nutrition Breakdown")
        macro_cols = st.columns(3)
        macros = nutrition["macros"]
        with macro_cols[0]:
            st.metric("Protein", f"{macros['protein']['value']:.1f}g", f"{macros['protein']['daily_value']}% DV")
        with macro_cols[1]:
            st.metric("Carbs", f"{macros['carbs']['value']:.1f}g", f"{macros['carbs']['daily_value']}% DV")
        with macro_cols[2]:
            st.metric("Fat", f"{macros['fat']['value']:.1f}g", f"{macros['fat']['daily_value']}% DV")
    if analysis_data["components"]:
        st.markdown("### 🍱 Meal Components")
        for item in analysis_data["components"]:
            with st.expander(f"{item['name']} ({item['type']})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Calories", f"{item['nutrition']['calories']} kcal")
                    st.metric("Protein", f"{item['nutrition']['protein']:.1f}g")
                with col2:
                    st.metric("Carbs", f"{item['nutrition']['carbs']:.1f}g")
                    st.metric("Fat", f"{item['nutrition']['fat']:.1f}g")

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
        st.session_state.prefs = {'goal': 'Balanced', 'diet': 'None', 'health_conditions': [], 'calorie_target': 2500}
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
        st.session_state.prefs.update({'goal': goal, 'diet': diet, 'health_conditions': health_conditions, 'calorie_target': cal_val})
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


def main():
    st.set_page_config(page_title="🍽️ Smart Dish Analyzer", page_icon="🍽️", layout="centered")
    load_styles()
    init_preferences()
    preferences_sidebar()
    st.markdown('<div class="topbar"><div class="title">🍽️ FitPlate AI</div><div class="theme-toggle">', unsafe_allow_html=True)
    theme_toggle()
    st.markdown('</div></div>', unsafe_allow_html=True)
    # Demo mode toggle and JSON path
    demo_default = bool(os.getenv('UX_DEMO') or os.getenv('UX_DATA_JSON'))
    with st.sidebar:
        st.markdown("### 🧪 Demo Mode")
        demo_mode = st.toggle("Run without API (JSON)", value=demo_default)
        json_path = st.text_input("Demo JSON path", value=os.getenv('UX_DATA_JSON', 'data/demo_meal.json')) if demo_mode else None
        
        st.markdown("### ⚡ Speed Settings")
        speed_mode = st.toggle("Fast Mode (Faster analysis, less detail)", value=True)
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
        prefs_str = json.dumps(st.session_state.prefs, sort_keys=True)
        prefs_hash = hashlib.md5(prefs_str.encode()).hexdigest()
        cache_key = f"{image_hash}_{prefs_hash}"
        
        # Check if we already have results for this image + preferences combination
        if 'analysis_cache' not in st.session_state:
            st.session_state.analysis_cache = {}
        
        # Initialize variables
        dish = None
        nutrition = None
        suggestions = []
        
        if cache_key in st.session_state.analysis_cache:
            # Use cached results (including suggestions that match current preferences)
            cached_data = st.session_state.analysis_cache[cache_key]
            dish = cached_data['dish']
            nutrition = cached_data['nutrition']
            suggestions = cached_data['suggestions']
        elif image_hash in [k.split('_')[0] for k in st.session_state.analysis_cache.keys()]:
            # We have analysis for this image but with different preferences
            # Reuse analysis but regenerate suggestions
            for key, cached_data in st.session_state.analysis_cache.items():
                if key.startswith(image_hash):
                    dish = cached_data['dish']
                    nutrition = cached_data['nutrition']
                    break
            
            # Generate new suggestions with current preferences
            if not demo_mode and nutrition is not None:
                with st.spinner("Updating suggestions for your preferences..."):
                    if analyzer is not None:
                        suggestions = safe_run_async(analyzer.suggest_improvements(nutrition, st.session_state.prefs))
                    else:
                        suggestions = []
            
            # Cache the new combination
            if dish is not None and nutrition is not None:
                st.session_state.analysis_cache[cache_key] = {
                    'dish': dish,
                    'nutrition': nutrition,
                    'suggestions': suggestions
                }
        else:
            # Perform analysis and cache results
            loading_placeholder = st.empty()
            with loading_placeholder.container():
                st.markdown('<div class="progress-line"><div class="bar"></div></div>', unsafe_allow_html=True)
                st.markdown('<div class="ai-analyzing mt-2"><i class="fa-solid fa-wand-magic-sparkles"></i> Analyzing <span class="ai-dots"><span></span><span></span><span></span></span></div>', unsafe_allow_html=True)
                st.markdown('<div class="card"><div class="skel-grid"><div class="skel-line shimmer" style="height:24px"></div><div class="skel-line shimmer" style="height:24px"></div><div class="skel-line shimmer" style="height:24px"></div></div></div>', unsafe_allow_html=True)
            if demo_mode:
                with st.spinner("Loading demo analysis..."):
                    if json_path is not None:
                        dish, nutrition, suggestions = _load_demo_from_json(json_path)
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
            loading_placeholder.empty()
            
            # Get AI suggestions (API) or use demo suggestions
            suggestions = []  # Initialize suggestions
            if demo_mode:
                # suggestions should be set from demo data above
                pass
            else:
                with st.spinner("Getting personalized suggestions..."):
                    if analyzer is not None:
                        suggestions = safe_run_async(analyzer.suggest_improvements(nutrition, st.session_state.prefs))
                    else:
                        suggestions = []
            
            # Cache the results including suggestions with the preference-aware cache key
            st.session_state.analysis_cache[cache_key] = {
                'dish': dish,
                'nutrition': nutrition,
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
                # Minimal inline style for modern chips
                st.markdown(
                    """
                    <style>
                      .chips{display:flex;flex-wrap:wrap;gap:.5rem;margin:.25rem 0 1rem;}
                      .chip{display:inline-flex;align-items:center;padding:.35rem .6rem;border-radius:999px;font-size:0.85rem;border:1px solid rgba(0,0,0,.08);}
                      .chip-warn{background:rgba(255, 59, 48, .08); border-color: rgba(255,59,48,.25);} /* red */
                      .chip-sugg{background:rgba(52, 199, 89, .08); border-color: rgba(52,199,89,.25);} /* green */
                      .chip .ico{margin-right:.4rem}
                      .section-title{font-weight:600;margin:0 0 .5rem 0;display:flex;align-items:center;gap:.5rem;}
                      .meal-title-row{display:flex;align-items:center;justify-content:space-between;gap:.75rem}
                      .meal-title{font-size:1.2rem;font-weight:800;display:flex;align-items:center;gap:.5rem}
                      .meal-badge{border-radius:999px;padding:.25rem .6rem;background:rgba(99,102,241,.12);border:1px solid rgba(99,102,241,.25);font-weight:700}
                      .meal-meta{display:flex;gap:1rem;color:rgba(0,0,0,.7);font-size:.92rem;margin-top:.35rem}
                      .diet-chips{display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.5rem}
                      .diet-chip{background:rgba(0,0,0,.06);border:1px solid rgba(0,0,0,.08);border-radius:999px;padding:.25rem .55rem;font-size:.8rem}
                                            /* Professional spacing and stat card */
                                            .section-space{margin-top:1rem}
                                            .section-space-lg{margin-top:1.25rem}
                                            .stat-card{margin-top:1rem;padding:1rem;border-radius:14px;background:linear-gradient(180deg, rgba(99,102,241,.10), rgba(99,102,241,.04));border:1px solid rgba(99,102,241,.25);box-shadow:0 2px 12px rgba(0,0,0,.06)}
                                            .stat-label{font-size:.8rem;color:rgba(0,0,0,.6);letter-spacing:.02em}
                                            .stat-value{font-size:2rem;font-weight:900;display:flex;align-items:baseline;gap:.35rem;margin:.35rem 0}
                                            .stat-value span{font-size:1rem;color:rgba(0,0,0,.65);font-weight:600}
                                            .stat-meter{height:8px;background:rgba(0,0,0,.08);border-radius:8px;overflow:hidden;margin-top:.25rem}
                                            .stat-meter .fill{height:100%;background:linear-gradient(90deg, #6366F1, #22D3EE);}
                                            .stat-sub{font-size:.85rem;color:rgba(0,0,0,.65);margin-top:.45rem}
                                            /* Card container for stacked sections */
                                            .ui-card{background:rgba(255,255,255,.85);backdrop-filter:saturate(180%) blur(6px);border:1px solid rgba(0,0,0,.08);border-radius:16px;padding:1rem;box-shadow:0 6px 20px rgba(0,0,0,.06);margin-bottom:1rem}
                                            .ui-card-title{font-weight:750;display:flex;align-items:center;gap:.5rem;margin-bottom:.5rem;font-size:1rem}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                # Simplified meal card - just name, confidence, and calorie bar
                cal_val = nutrition_summary["calories"]["value"]
                cal_dv = nutrition_summary["calories"]["daily_value"]
                title_html = f"<div class='meal-title-row'><div class='meal-title'>{confidence_emoji} {dish['dish_name']}</div><span class='meal-badge'>{confidence:.1f}%</span></div>"
                
                meal_card_html = f"""
                    <div class='ui-card'>
                        {title_html}
                        <div class='stat-value'>{cal_val} <span>kcal</span></div>
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

                # Show AI Suggestions and Warnings below the Dish info
                if suggestions:
                    sugg_chips = [
                        f"<span class='chip chip-sugg'><span class='ico'>{_sugg_icon(s)}</span>{s}</span>"
                        for s in suggestions
                    ]
                    st.markdown(
                        f"<div class='chips'>{''.join(sugg_chips)}</div>",
                        unsafe_allow_html=True,
                    )

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
                # Constrain image height for above-the-fold layout
                st.markdown(
                    """
                    <style>
                      .image-preview img{max-height:320px;object-fit:cover;border-radius:12px}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(image, caption="Your Dish", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Row: Left = Additional Nutrients, Right = Macronutrients (aligned with image width and equal height)
        with st.container():
            st.markdown(
                """
                <style>
                  .ui-card.eq{min-height:220px;display:flex;flex-direction:column}
                </style>
                """,
                unsafe_allow_html=True,
            )
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
            
            cols = st.columns(len(chat_suggestions))
            preset = None
            for i, (col, s) in enumerate(zip(cols, chat_suggestions)):
                with col:
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
