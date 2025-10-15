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

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nest_asyncio.apply()

load_dotenv()
CLIENT_AVAILABLE = False
client: Optional[AsyncOpenAI] = None

from app.config import VISION_MODEL, CHAT_MODEL

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
    try:
        _ts = datetime.utcnow().isoformat()
        _start = time.perf_counter()
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
        _dur_ms = int((time.perf_counter() - _start) * 1000) if '_start' in locals() else 0
        logger.error(f"[Chat][ERROR] dur_ms={_dur_ms} error={str(e)}")
        yield f"Sorry, I encountered an error: {str(e)}"

def generate_chat_suggestions(user_preferences: Dict[str, Any]) -> List[str]:
    """Generate personalized chat suggestions based on user preferences"""
    suggestions = []
    
    goal = user_preferences.get('goal', 'Balanced')
    diet = user_preferences.get('diet', 'None')
    health_conditions = user_preferences.get('health_conditions', [])
    calorie_target = user_preferences.get('calorie_target', 2500)
    
    # Goal-adapted questions
    if goal == 'Bulking':
        suggestions.extend([
            "Does this meal provide enough protein for bulking?",
            "How can I add more calories to support my bulking goals?"
        ])
    elif goal == 'Cutting':
        suggestions.extend([
            "How can I reduce calories while keeping this meal satisfying?",
            f"Is this meal appropriate for my daily calorie target of {calorie_target} kcal?"
        ])
    elif goal == 'Maintenance':
        suggestions.extend([
            "Does this meal provide enough protein for maintenance?",
            f"Is this meal appropriate for my daily calorie target of {calorie_target} kcal?"
        ])
    else:  # Balanced
        suggestions.extend([
            "Is this a well-balanced meal for my goals?",
            "Can I add more vegetables or fiber to improve this meal?"
        ])
    
    # Diet-adapted questions
    if diet != 'None':
        suggestions.append(f"Is this meal suitable for a {diet} diet?")
        if diet in ['Keto', 'Vegan', 'Vegetarian', 'Mediterranean']:
            suggestions.append(f"Which ingredients should I replace to make this meal compliant with a {diet} diet?")
    
    # Health condition-adapted questions
    if health_conditions:
        # Handle multiple conditions - pick the most relevant ones
        primary_condition = health_conditions[0]
        suggestions.append(f"Is this meal suitable for someone with {primary_condition}?")
        
        # Add specific allergen questions based on selected conditions
        if 'Celiac Disease' in health_conditions:
            suggestions.append("Does this meal contain gluten?")
        elif 'Lactose Intolerance' in health_conditions:
            suggestions.append("Does this meal contain lactose?")
        elif len(health_conditions) > 1:
            # If multiple conditions but no specific allergen ones, ask about all conditions
            conditions_str = ", ".join(health_conditions[:2])  # Limit to first 2 for readability
            suggestions.append(f"Is this meal safe for my health conditions ({conditions_str})?")
    
    # Always include some general options
    suggestions.extend([
        "Which ingredients should I reduce or replace to make this meal healthier?",
        "What are the main nutritional benefits of this meal?"
    ])
    
    # Return max 3 suggestions to fit the UI nicely
    return suggestions[:3]

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

    # Initialize OpenAI client and analyzer only if not in demo mode
    analyzer: Optional[FoodAnalyzerService] = None
    if not demo_mode:
        loop = asyncio.get_event_loop()
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

        access_ok, error_msg = loop.run_until_complete(verify_openai_access(client))
        if not access_ok:
            st.error(f"⚠️ OpenAI API Error: {error_msg}")
            st.info("Turn on Demo Mode in the sidebar to run without API access.")
            return
        analyzer = FoodAnalyzerService(api_key=api_key)
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
        # Create a unique key for this image
        import hashlib
        image_hash = hashlib.md5(selected_bytes).hexdigest()
        
        # Check if we already have results for this image
        if 'analysis_cache' not in st.session_state:
            st.session_state.analysis_cache = {}
        
        if image_hash in st.session_state.analysis_cache:
            # Use cached results
            cached_data = st.session_state.analysis_cache[image_hash]
            dish = cached_data['dish']
            nutrition = cached_data['nutrition']
            
            # Check if suggestions exist in cache (for backward compatibility)
            if 'suggestions' in cached_data:
                suggestions = cached_data['suggestions']
            else:
                # Old cache format - fetch suggestions now
                if demo_mode:
                    suggestions = []
                else:
                    with st.spinner("Getting personalized suggestions..."):
                        suggestions = loop.run_until_complete(analyzer.suggest_improvements(nutrition))
                # Update cache with suggestions
                st.session_state.analysis_cache[image_hash]['suggestions'] = suggestions
        else:
            # Perform analysis and cache results
            loading_placeholder = st.empty()
            with loading_placeholder.container():
                st.markdown('<div class="progress-line"><div class="bar"></div></div>', unsafe_allow_html=True)
                st.markdown('<div class="ai-analyzing mt-2"><i class="fa-solid fa-wand-magic-sparkles"></i> Analyzing <span class="ai-dots"><span></span><span></span><span></span></span></div>', unsafe_allow_html=True)
                st.markdown('<div class="card"><div class="skel-grid"><div class="skel-line shimmer" style="height:24px"></div><div class="skel-line shimmer" style="height:24px"></div><div class="skel-line shimmer" style="height:24px"></div></div></div>', unsafe_allow_html=True)
            if demo_mode:
                with st.spinner("Loading demo analysis..."):
                    dish, nutrition, suggestions = _load_demo_from_json(json_path)
            else:
                with st.spinner("Analyzing your dish..."):
                    loop = asyncio.get_event_loop()
                    dish, nutrition = loop.run_until_complete(process_image(image, analyzer, st.session_state.prefs))
            loading_placeholder.empty()
            
            # Get AI suggestions (API) or use demo suggestions
            if demo_mode:
                suggestions = suggestions if 'suggestions' in locals() else []
            else:
                with st.spinner("Getting personalized suggestions..."):
                    suggestions = loop.run_until_complete(analyzer.suggest_improvements(nutrition, st.session_state.prefs))
            
            # Cache the results including suggestions
            st.session_state.analysis_cache[image_hash] = {
                'dish': dish,
                'nutrition': nutrition,
                'suggestions': suggestions
            }
        
        # Check if this is a non-food item
        if dish.get('is_non_food', False) or dish['dish_name'] == "Non-Food Item":
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
        
        # Prepare common values for the left info column
        nutrition_summary = nutrition["nutrition_summary"]
        meal_info = nutrition.get("meal_info", {})
        confidence = dish['confidence'] * 100
        confidence_emoji = "🎯" if confidence >= 90 else "✨" if confidence >= 70 else "👀"
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

                # Meal header + calories (as a card)
                cal_val = nutrition_summary["calories"]["value"]
                cal_dv = nutrition_summary["calories"]["daily_value"]
                serving_size = meal_info.get("serving_size", "N/A")
                title_html = f"<div class='meal-title-row'><div class='meal-title'>{confidence_emoji} {dish['dish_name']}</div><span class='meal-badge'>{confidence:.1f}%</span></div>"
                meta_html = f"<div class='meal-meta'>🔥 {cal_val} kcal • {cal_dv}% DV · 📏 Serving: {serving_size}</div>"
                tags_html = ""
                if diet_tags:
                    tags_html = "<div class='diet-chips'>" + " ".join([f"<span class='diet-chip'>🏷️ {tag}</span>" for tag in diet_tags]) + "</div>"
                meal_card_html = f"""
                    <div class='ui-card'>
                        {title_html}
                        {meta_html}
                        {tags_html}
                        <div class='stat-card'>
                          <div class='stat-label'>Total Calories</div>
                          <div class='stat-value'>{cal_val} <span>kcal</span></div>
                          <div class='stat-meter'><div class='fill' style='width:{cal_dv}%' /></div>
                          <div class='stat-sub'>{cal_dv}% Daily Value • Serving: {serving_size}</div>
                        </div>
                    </div>
                """
                st.markdown(meal_card_html, unsafe_allow_html=True)

                # Icon helpers based on common nutrition keywords
                def _warn_icon(txt: str) -> str:
                    t = txt.lower()
                    if "sodium" in t or "salt" in t: return "🧂"
                    if "sugar" in t: return "🍬"
                    if "fat" in t and "saturated" in t: return "🧈"
                    if "fat" in t: return "🥓"
                    if "calorie" in t: return "🔥"
                    if "fiber" in t: return "🌾"
                    if "protein" in t: return "🥩"
                    if "carb" in t: return "🍞"
                    return "⚠️"

                def _sugg_icon(txt: str) -> str:
                    t = txt.lower()
                    if "vegetable" in t or "veggie" in t or "broccoli" in t or "salad" in t: return "🥦"
                    if "fruit" in t: return "🍎"
                    if "protein" in t or "chicken" in t: return "🥩"
                    if "fiber" in t or "whole" in t or "quinoa" in t: return "🌾"
                    if "fat" in t and ("olive" in t or "avocado" in t): return "🫒"
                    if "sugar" in t: return "🍬"
                    if "water" in t or "hydrate" in t: return "💧"
                    if "calorie" in t or "reduce" in t: return "⚡"
                    return "💡"

                # Show AI Suggestions and Warnings below the Dish info
                if suggestions:
                    sugg_chips = [
                        f"<span class='chip chip-sugg'><span class='ico'>{_sugg_icon(s)}</span>{s}</span>"
                        for s in suggestions
                    ]
                    st.markdown(
                        f"<div class='ui-card'><div class='ui-card-title'>💡 AI Suggestions</div><div class='chips'>{''.join(sugg_chips)}</div></div>",
                        unsafe_allow_html=True,
                    )

                warn_list = nutrition_summary.get("warnings", [])
                if warn_list:
                    warn_chips = [
                        f"<span class='chip chip-warn'><span class='ico'>{_warn_icon(w)}</span>{w}</span>"
                        for w in warn_list
                    ]
                    st.markdown(
                        f"<div class='ui-card'><div class='ui-card-title'>⚠️ Warnings</div><div class='chips'>{''.join(warn_chips)}</div></div>",
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
                    f"<span class='chip'><span class='ico'>🌾</span>Fiber: {fiber.get('value', 0):.1f}g • {fiber.get('daily_value', 0)}% DV</span>",
                    f"<span class='chip'><span class='ico'>🍯</span>Sugar: {sugar.get('value', 0):.1f}g</span>",
                    f"<span class='chip'><span class='ico'>🥑</span>Saturated Fat: {sat.get('value', 0):.1f}g • {sat.get('daily_value', 0)}% DV</span>",
                ]
                st.markdown(
                    f"<div class='ui-card eq'><div class='ui-card-title'>➕ Additional Nutrients</div><div class='chips'>{''.join(add_chips)}</div></div>",
                    unsafe_allow_html=True,
                )
            with col_right:
                macros = nutrition_summary["macros"]
                macro_html = f"""
                    <div class='ui-card eq'>
                        <div class='ui-card-title'>📊 Macronutrients</div>
                        <div class='chips'>
                          <span class='chip'><span class='ico'>🥩</span>Protein: {macros['protein']['value']:.1f}g • {macros['protein']['daily_value']}% DV</span>
                          <span class='chip'><span class='ico'>🍚</span>Carbs: {macros['carbs']['value']:.1f}g • {macros['carbs']['daily_value']}% DV</span>
                          <span class='chip'><span class='ico'>🫒</span>Fat: {macros['fat']['value']:.1f}g • {macros['fat']['daily_value']}% DV</span>
                        </div>
                    </div>
                """
                st.markdown(macro_html, unsafe_allow_html=True)

        
        # Macronutrients are shown next to Additional Nutrients above to align with the image column
        components = nutrition.get("components", [])
        if components:
            with st.expander("🍱 Meal Components", expanded=False):
                for item in components:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    type_icons = {"main_dish":"🍽️","side_dish":"🥗","beverage":"🥤","condiment":"🧂"}
                    icon = type_icons.get(item['type'], "🍴")
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
            
            # Generate personalized chat suggestions
            chat_suggestions = generate_chat_suggestions(st.session_state.prefs)
            
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
                        async for token in handle_chat(analyzer, user_question, nutrition, st.session_state.prefs):
                            yield token
                    st.write_stream(stream_answer())
        elif nutrition and demo_mode:
            st.markdown('<hr/>', unsafe_allow_html=True)
            st.info("Chat is disabled in Demo Mode. Disable Demo Mode in the sidebar to ask questions.")

if __name__ == "__main__":
    main()
