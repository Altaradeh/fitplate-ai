import asyncio
import json
import logging
import os
import re
import sys
import argparse
from typing import Optional, Tuple, List, Dict, Any

import nest_asyncio
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from services.analyzer import FoodAnalyzerService

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

load_dotenv()

# Optional demo mode using a local JSON file (no API calls)
def _parse_args():
    try:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--data-json", dest="data_json", default=None)
        args, _ = parser.parse_known_args(sys.argv[1:])
        return args
    except Exception:
        class _A: data_json=None
        return _A()

_args = _parse_args()
DEMO_JSON_PATH = os.getenv("UX_DATA_JSON") or (_args.data_json if hasattr(_args, "data_json") else None)
DEMO_MODE = bool(DEMO_JSON_PATH)
DEMO_DATA: Optional[Dict[str, Any]] = None
if DEMO_MODE:
    try:
        with open(DEMO_JSON_PATH, "r", encoding="utf-8") as f:
            DEMO_DATA = json.load(f)
    except Exception as e:
        DEMO_DATA = None
        DEMO_MODE = False
        logging.getLogger(__name__).error(f"Failed to load demo JSON: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nest_asyncio.apply()

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_key")

# Only set up OpenAI client/analyzer when not in demo mode
client = None
OpenAIError = Exception  # fallback type
analyzer = None
if not DEMO_MODE:
    try:
        from openai import AsyncOpenAI, OpenAIError as _OpenAIError  # type: ignore
        OpenAIError = _OpenAIError
        if not api_key:
            raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY in your .env file")
        client = AsyncOpenAI(api_key=api_key)
        analyzer = FoodAnalyzerService(api_key=api_key)
    except Exception as e:
        # If anything fails, remain without client and handle later
        logging.getLogger(__name__).error(f"OpenAI setup failed: {e}")

from app.config import VISION_MODEL, CHAT_MODEL

async def verify_openai_access() -> Tuple[bool, Optional[str]]:
    if DEMO_MODE:
        # In demo mode we explicitly avoid any network calls
        logger.info("Demo mode enabled: skipping OpenAI verification")
        return True, None
    try:
        if client is None:
            return False, "OpenAI client not initialized"
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

async def process_image(image):
    try:
        if DEMO_MODE and DEMO_DATA:
            meal = DEMO_DATA.get("meal_info", {})
            dish_name = meal.get("name", "Unknown Dish")
            confidence = meal.get("confidence", 0.0)
            return {"dish_name": dish_name, "confidence": confidence}, DEMO_DATA

        if analyzer is None:
            raise RuntimeError("Analyzer not available")
        analysis_result = await analyzer.analyze_meal(image)
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
        if dish_name == "Unknown Dish" and components:
            dish_name = components[0].get("name", "Unknown Dish")
        logger.info(f"Successfully analyzed: {dish_name} (confidence: {confidence:.2f})")
        return {
            "dish_name": dish_name,
            "confidence": confidence
        }, analysis_result
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {
            "dish_name": "Unknown Dish",
            "confidence": 0.0
        }, None

async def handle_chat(analyzer: FoodAnalyzerService, question: str, analysis_result: Dict[str, Any]):
    import time
    from datetime import datetime
    try:
        _ts = datetime.utcnow().isoformat()
        _start = time.perf_counter()
        logger.info(f"[Chat][START] ts={_ts} question_len={len(question)}")
        if DEMO_MODE:
            async def _demo_stream():
                msg = (
                    "✅ SUMMARY: Looks balanced with solid protein.\n\n"
                    "📘 DETAILS: Consider more veggies and keep sauces light."
                )
                for i in range(0, len(msg), 24):
                    yield msg[i:i+24]
                    await asyncio.sleep(0.02)
            stream = _demo_stream()
        else:
            stream = await analyzer.answer_question(question, analysis_result)
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
        with st.spinner("Getting personalized suggestions..."):
            suggestions = loop.run_until_complete(analyzer.suggest_improvements(analysis_data))
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
        st.session_state.prefs = {'goal': 'balanced', 'diet': 'none', 'calorie_target': None}
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
        goal = st.selectbox("Goal", ["balanced", "bulking", "cutting", "maintenance"], index=["balanced","bulking","cutting","maintenance"].index(st.session_state.prefs.get('goal','balanced')))
        diet = st.selectbox("Diet", ["none", "keto", "vegan", "vegetarian", "mediterranean"], index=["none","keto","vegan","vegetarian","mediterranean"].index(st.session_state.prefs.get('diet','none')))
        cal = st.text_input("Daily calorie target (optional)", value=str(st.session_state.prefs.get('calorie_target') or ''))
        cal_val = None
        try:
            cal_val = int(cal) if cal.strip() else None
        except Exception:
            pass
        st.session_state.prefs.update({'goal': goal, 'diet': diet, 'calorie_target': cal_val})
        st.markdown(
            f"""
            <script>
            try {{
              const prefs = {json.dumps({'goal': goal, 'diet': diet, 'calorie_target': cal_val})};
              window.localStorage.setItem('fitplate-prefs', JSON.stringify(prefs));
            }} catch(e) {{}}
            </script>
            """,
            unsafe_allow_html=True,
        )

def main():
    st.set_page_config(page_title="🍽️ Smart Dish Analyzer", page_icon="🍽️", layout="centered")
    load_styles()
    init_preferences()
    preferences_sidebar()
    st.markdown('<div class="topbar"><div class="title">🍽️ FitPlate AI</div><div class="theme-toggle">', unsafe_allow_html=True)
    theme_toggle()
    st.markdown('</div></div>', unsafe_allow_html=True)
    loop = asyncio.get_event_loop()
    access_ok, error_msg = loop.run_until_complete(verify_openai_access())
    if not access_ok:
        st.error(f"⚠️ OpenAI API Error: {error_msg}")
        st.info("Please check your OpenAI API key in the .env file and ensure you have access to GPT-4 Vision API.")
        return
    if DEMO_MODE:
        st.info(f"Using local demo data: {DEMO_JSON_PATH}")
    st.write("Upload a meal photo or use your camera to analyze your food and chat about your goals.")
    tab_upload, tab_camera = st.tabs(["📂 Upload Photo", "📷 Take Photo"]) 
    image = None
    with tab_upload:
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
    with tab_camera:
        camera_input = st.camera_input("Take a photo")
        if camera_input:
            image = Image.open(camera_input)
    if image:
        with st.container():
            st.markdown('<div class="card image-preview">', unsafe_allow_html=True)
            st.image(image, caption="Your Dish", width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Use session state to cache analysis results and avoid re-analyzing on chat interactions
        # Create a unique key for this image
        import hashlib
        image_bytes = image.tobytes()
        image_hash = hashlib.md5(image_bytes).hexdigest()
        
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
                if DEMO_MODE and nutrition and nutrition.get('ai_insights'):
                    insights = nutrition.get('ai_insights', {})
                    suggestions = insights.get('improvements') or insights.get('suggestions') or []
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
            with st.spinner("Analyzing your dish..."):
                loop = asyncio.get_event_loop()
                dish, nutrition = loop.run_until_complete(process_image(image))
            loading_placeholder.empty()
            
            # Get AI suggestions: in demo mode use local data; otherwise call service
            if DEMO_MODE and nutrition and nutrition.get('ai_insights'):
                insights = nutrition.get('ai_insights', {})
                suggestions = insights.get('improvements') or insights.get('suggestions') or []
            else:
                with st.spinner("Getting personalized suggestions..."):
                    suggestions = loop.run_until_complete(analyzer.suggest_improvements(nutrition))
            
            # Cache the results including suggestions
            st.session_state.analysis_cache[image_hash] = {
                'dish': dish,
                'nutrition': nutrition,
                'suggestions': suggestions
            }
        
        # Always render the UI (using cached or fresh data)
        if not nutrition or not nutrition.get("nutrition_summary"):
            st.error("⚠️ Could not analyze nutrition information")
            return
        
        nutrition_summary = nutrition["nutrition_summary"]
        meal_info = nutrition.get("meal_info", {})
        # If running with demo JSON, ensure dish variable matches demo
        if DEMO_MODE and nutrition.get("meal_info"):
            dish = {
                "dish_name": nutrition["meal_info"].get("name", dish.get("dish_name", "Unknown Dish")),
                "confidence": nutrition["meal_info"].get("confidence", dish.get("confidence", 0.0)),
            }
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            confidence = dish['confidence'] * 100
            confidence_emoji = "🎯" if confidence >= 90 else "✨" if confidence >= 70 else "👀"
            st.markdown(f"<div class='section-header'>{confidence_emoji} {dish['dish_name']} <span class='chip'>{confidence:.1f}%</span></div>", unsafe_allow_html=True)
            cal_col, tags_col = st.columns([1, 2])
            with cal_col:
                cal_dv = nutrition_summary["calories"]["daily_value"]
                st.metric("Total Calories", f"{nutrition_summary['calories']['value']} kcal", f"{cal_dv}% Daily Value", help="Based on 2000 kcal daily requirement", delta_color="off")
                serving_size = meal_info.get("serving_size", "N/A")
                st.caption(f"📏 Serving: {serving_size}")
            with tags_col:
                diet_tags = nutrition_summary.get("diet_tags", [])
                if diet_tags:
                    tags_html = " ".join([f"<span class=\"diet-tag\">🏷️ {tag}</span>" for tag in diet_tags])
                    st.markdown(f"<div class='mt-2'>{tags_html}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="subsection-header">Macronutrients</div>', unsafe_allow_html=True)
            macro_cols = st.columns(3)
            macros = nutrition_summary["macros"]
            with macro_cols[0]:
                st.metric("🥩 Protein", f"{macros['protein']['value']:.1f}g", f"{macros['protein']['daily_value']}% DV")
            with macro_cols[1]:
                st.metric("🍚 Carbs", f"{macros['carbs']['value']:.1f}g", f"{macros['carbs']['daily_value']}% DV")
            with macro_cols[2]:
                st.metric("🫒 Fat", f"{macros['fat']['value']:.1f}g", f"{macros['fat']['daily_value']}% DV")
            st.markdown('</div>', unsafe_allow_html=True)
        with st.expander("Additional Nutrients"):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            detail_cols = st.columns(2)
            additional = nutrition_summary["additional"]
            with detail_cols[0]:
                st.metric("🌾 Fiber", f"{additional['fiber']['value']:.1f}g", f"{additional['fiber']['daily_value']}% DV")
                st.metric("🍯 Sugar", f"{additional['sugar']['value']:.1f}g", "No DV established")
            with detail_cols[1]:
                st.metric("🥑 Saturated Fat", f"{additional['saturated_fat']['value']:.1f}g", f"{additional['saturated_fat']['daily_value']}% DV")
            st.markdown('</div>', unsafe_allow_html=True)
        components = nutrition.get("components", [])
        if components:
            st.markdown('<div class="section-header">🍱 Meal Components</div>', unsafe_allow_html=True)
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
        colw, cols = st.columns(2)
        with colw:
            warnings = nutrition_summary.get("warnings", [])
            if warnings:
                with st.expander("⚠️ Warnings", expanded=False):
                    for warning in warnings:
                        st.warning(warning, icon="⚠️")
        with cols:
            with st.expander("💡 AI Suggestions", expanded=False):
                # Use cached suggestions (already fetched during analysis)
                for tip in suggestions:
                    st.write(tip)
        
        # Chat section - always available after analysis (both cached and fresh)
        if nutrition:
            st.markdown('<hr/>', unsafe_allow_html=True)
            st.markdown('<div class="subsection-header">Ask about your meal</div>', unsafe_allow_html=True)
            chat_suggestions = ["Is this good for bulking?", "How to reduce calories?", "Is this balanced for dinner?"]
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
                        async for token in handle_chat(analyzer, user_question, nutrition):
                            yield token
                    st.write_stream(stream_answer())

if __name__ == "__main__":
    main()
