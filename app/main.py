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

from services.analyzer import FoodAnalyzerService

# Create async loop for handling async operations
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Load environment variables
load_dotenv()

# Initialize analyzer service with API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
analyzer = FoodAnalyzerService(api_key=api_key)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply nest_asyncio to make async work in Streamlit
nest_asyncio.apply()

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_key")
if not api_key:
    raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY in your .env file")

# Initialize OpenAI client
client = AsyncOpenAI(api_key=api_key)

# Model configuration
from app.config import VISION_MODEL, CHAT_MODEL

async def verify_openai_access() -> Tuple[bool, Optional[str]]:
    """Verify OpenAI API access and model availability."""
    try:
        # Test with a simple completion
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
    """Process an uploaded image using the analyzer service."""
    try:
        # Use the analyzer service to analyze the meal
        analysis_result = await analyzer.analyze_meal(image)
        
        if not analysis_result or analysis_result.get("status") == "error":
            error_msg = analysis_result.get("error", "Unknown error") if analysis_result else "No result"
            logger.error(f"Analysis failed: {error_msg}")
            return {
                "dish_name": "Unknown Dish",
                "confidence": 0.0
            }, None

        # Extract meal info from the new structure
        meal_info = analysis_result.get("meal_info", {})
        components = analysis_result.get("components", [])
        
        dish_name = meal_info.get("name", "Unknown Dish")
        confidence = meal_info.get("confidence", 0.0)
        
        # If no meal_info but we have components, use the first component
        if dish_name == "Unknown Dish" and components:
            dish_name = components[0].get("name", "Unknown Dish")
            
        logger.info(f"Successfully analyzed: {dish_name} (confidence: {confidence:.2f})")
        
        # Return the dish info and the full analysis result
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
    """Handle the chat interaction using the analyzer service."""
    try:
        stream = await analyzer.answer_question(question, analysis_result)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        yield f"Sorry, I encountered an error: {str(e)}"



def load_styles():
    """Load Google Fonts, FontAwesome, and external CSS with theme hook."""
    # Google Fonts + FontAwesome
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" crossorigin="anonymous" referrerpolicy="no-referrer" />
        """,
        unsafe_allow_html=True,
    )

    # Persisted theme from localStorage; fallback to system preference
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

    # Load external CSS
    try:
      with open(os.path.join(os.path.dirname(__file__), 'styles.css')) as f:
          st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
      st.warning(f"Could not load styles.css: {e}")

def render_meal_analysis(analysis_data):
    """Render the meal analysis UI."""
    if analysis_data["status"] == "error":
        st.error(f"Analysis failed: {analysis_data['error']}")
        return

    # Meal Overview
    with st.container():
        st.title("🍽️ Meal Analysis")
        meal = analysis_data["meal_info"]
        st.success(f"📸 Analyzed: {meal['name']} ({meal['confidence']:.1f}% confidence)")

    # Main Nutrition Card
    with st.container():
        col1, col2 = st.columns([2, 3])
        
        # Calories and serving
        with col1:
            nutrition = analysis_data["nutrition_summary"]
            st.metric(
                "Total Calories",
                f"{nutrition['calories']['value']} kcal",
                f"{nutrition['calories']['daily_value']}% DV"
            )
            st.caption(f"Serving: {meal['serving_size']}")
        
        # Diet tags
        with col2:
            if nutrition["diet_tags"]:
                tags_html = " ".join([
                    f'<span class="nutrient-tag">{tag}</span>'
                    for tag in nutrition["diet_tags"]
                ])
                st.markdown(f"<div style='margin-top: 0.5rem'>{tags_html}</div>", unsafe_allow_html=True)

    # AI Insights
    insights = analysis_data["ai_insights"]
    with st.container():
        st.markdown("### 🤖 AI Insights")
        
        # Health Score
        score_color = "green" if insights["health_score"] >= 80 else "orange" if insights["health_score"] >= 60 else "red"
        st.markdown(f"**Health Score:** <span style='color: {score_color}'>{insights['health_score']}/100</span>", unsafe_allow_html=True)
        
            # AI-powered recommendations
        with st.spinner("Getting personalized suggestions..."):
            suggestions = loop.run_until_complete(analyzer.suggest_improvements(analysis_data))
            for suggestion in suggestions:
                st.markdown(suggestion)        # Dietary Considerations
        if insights["dietary_considerations"]:
            with st.expander("📋 Dietary Considerations"):
                for consideration in insights["dietary_considerations"]:
                    st.markdown(f"• {consideration}")

    # Nutrition Details
    with st.container():
        st.markdown("### 📊 Nutrition Breakdown")
        
        # Macronutrients
        macro_cols = st.columns(3)
        macros = nutrition["macros"]
        
        with macro_cols[0]:
            st.metric("Protein", f"{macros['protein']['value']:.1f}g", f"{macros['protein']['daily_value']}% DV")
        with macro_cols[1]:
            st.metric("Carbs", f"{macros['carbs']['value']:.1f}g", f"{macros['carbs']['daily_value']}% DV")
        with macro_cols[2]:
            st.metric("Fat", f"{macros['fat']['value']:.1f}g", f"{macros['fat']['daily_value']}% DV")

    # Meal Components
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
    """Render theme toggle and persist choice in localStorage + session_state."""
    if 'theme' not in st.session_state:
        st.session_state.theme = None
    # Read theme from query param or session
    current = st.session_state.theme
    if not current:
        current = 'dark'  # default visually
    colA, colB = st.columns([6,1])
    with colB:
        toggle = st.toggle("Dark Mode", value=(current=='dark'), label_visibility="collapsed")
    theme = 'dark' if toggle else 'light'
    st.session_state.theme = theme
    # Apply to document
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
    """Initialize and hydrate user preferences from localStorage via JS bridge."""
    if 'prefs' not in st.session_state:
        st.session_state.prefs = {
            'goal': 'balanced',
            'diet': 'none',
            'calorie_target': None,
        }
    # Read from localStorage and update session_state (best-effort)
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
    """Render preferences in sidebar and persist changes."""
    with st.sidebar:
        st.markdown("### 🎯 Preferences")
        goal = st.selectbox(
            "Goal",
            ["balanced", "bulking", "cutting", "maintenance"],
            index=["balanced","bulking","cutting","maintenance"].index(st.session_state.prefs.get('goal','balanced')),
        )
        diet = st.selectbox(
            "Diet",
            ["none", "keto", "vegan", "vegetarian", "mediterranean"],
            index=["none","keto","vegan","vegetarian","mediterranean"].index(st.session_state.prefs.get('diet','none')),
        )
        cal = st.text_input("Daily calorie target (optional)", value=str(st.session_state.prefs.get('calorie_target') or ''))
        cal_val = None
        try:
            cal_val = int(cal) if cal.strip() else None
        except Exception:
            pass
        st.session_state.prefs.update({'goal': goal, 'diet': diet, 'calorie_target': cal_val})
        # Persist to localStorage
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
    """Main application entry point."""
    # Configure the page
    st.set_page_config(
        page_title="🍽️ Smart Dish Analyzer",
        page_icon="🍽️",
        layout="centered"
    )
    
    # Load custom styles and theme
    load_styles()

    # Hydrate preferences and render sidebar
    init_preferences()
    preferences_sidebar()

    # Top bar with title and theme toggle
    st.markdown('<div class="topbar">\
      <div class="title">🍽️ FitPlate AI</div>\
      <div class="theme-toggle">', unsafe_allow_html=True)
    theme_toggle()
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Verify OpenAI access
    loop = asyncio.get_event_loop()
    access_ok, error_msg = loop.run_until_complete(verify_openai_access())
    if not access_ok:
        st.error(f"⚠️ OpenAI API Error: {error_msg}")
        st.info("Please check your OpenAI API key in the .env file and ensure you have access to GPT-4 Vision API.")
        return
        
    st.write("Upload a meal photo or use your camera to analyze your food and chat about your goals.")

    # Input methods
    tab_upload, tab_camera = st.tabs(["📂 Upload Photo", "📷 Take Photo"]) 
    image = None
    with tab_upload:
        # Maintain uploader but rely on global CSS for visuals
        uploaded_file = st.file_uploader("Upload a dish photo", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
    with tab_camera:
        camera_input = st.camera_input("Take a photo")
        if camera_input:
            image = Image.open(camera_input)

    if image:
        # Preview
        with st.container():
            st.markdown('<div class="card image-preview">', unsafe_allow_html=True)
            st.image(image, caption="Your Dish", width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

        # Progress indicator and skeletons
        st.markdown('<div class="progress-line"><div class="bar"></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="ai-analyzing mt-2"><i class="fa-solid fa-wand-magic-sparkles"></i> Analyzing <span class="ai-dots"><span></span><span></span><span></span></span></div>', unsafe_allow_html=True)
        # Skeleton cards
        st.markdown('<div class="card"><div class="skel-grid">\
          <div class="skel-line shimmer" style="height:24px"></div>\
          <div class="skel-line shimmer" style="height:24px"></div>\
          <div class="skel-line shimmer" style="height:24px"></div>\
        </div></div>', unsafe_allow_html=True)

        with st.spinner("Analyzing your dish..."):
            loop = asyncio.get_event_loop()
            dish, nutrition = loop.run_until_complete(process_image(image))

        # Analysis container
        if not nutrition or not nutrition.get("nutrition_summary"):
            st.error("⚠️ Could not analyze nutrition information")
            return

        nutrition_summary = nutrition["nutrition_summary"]
        meal_info = nutrition.get("meal_info", {})

        # Overview card
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            confidence = dish['confidence'] * 100
            confidence_emoji = "🎯" if confidence >= 90 else "✨" if confidence >= 70 else "👀"
            st.markdown(f"<div class='section-header'>{confidence_emoji} {dish['dish_name']} <span class='chip'>{confidence:.1f}%</span></div>", unsafe_allow_html=True)
            cal_col, tags_col = st.columns([1, 2])
            with cal_col:
                cal_dv = nutrition_summary["calories"]["daily_value"]
                st.metric(
                    "Total Calories",
                    f"{nutrition_summary['calories']['value']} kcal",
                    f"{cal_dv}% Daily Value",
                    help="Based on 2000 kcal daily requirement",
                    delta_color="off"
                )
                serving_size = meal_info.get("serving_size", "N/A")
                st.caption(f"📏 Serving: {serving_size}")
            with tags_col:
                diet_tags = nutrition_summary.get("diet_tags", [])
                if diet_tags:
                    tags_html = " ".join([f"<span class=\"diet-tag\">🏷️ {tag}</span>" for tag in diet_tags])
                    st.markdown(f"<div class='mt-2'>{tags_html}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Macros card
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

        # Additional nutrients in collapsible section
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

        # Components list
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

        # Warnings & AI Suggestions
        colw, cols = st.columns(2)
        with colw:
            warnings = nutrition_summary.get("warnings", [])
            if warnings:
                with st.expander("⚠️ Warnings", expanded=False):
                    for warning in warnings:
                        st.warning(warning, icon="⚠️")
        with cols:
            with st.expander("💡 AI Suggestions", expanded=False):
                with st.spinner("Getting personalized suggestions..."):
                    tips = loop.run_until_complete(analyzer.suggest_improvements(nutrition))
                    for tip in tips:
                        st.write(tip)

        # Chat section - dock suggestions and input
        st.markdown('<hr/>', unsafe_allow_html=True)
        st.markdown('<div class="subsection-header">Ask about your meal</div>', unsafe_allow_html=True)
        # Suggestion buttons styled as chips
        suggestions = [
            "Is this good for bulking?",
            "How to reduce calories?",
            "Is this balanced for dinner?",
        ]
        cols = st.columns(len(suggestions))
        preset = None
        for i, (col, s) in enumerate(zip(cols, suggestions)):
            with col:
                if st.button(s, key=f"suggest_{i}"):
                    preset = s
        user_question = preset or st.chat_input("💬 Ask about your meal")
        if user_question:
            with st.container():
                st.chat_message("user", avatar="🤔").write(user_question)
                chat_box = st.chat_message("assistant", avatar="🍽️")
                response_placeholder = chat_box.empty()
            async def stream_response():
                response_text = ""
                try:
                    async for token in handle_chat(analyzer, user_question, nutrition):
                        response_text += token
                        response_placeholder.markdown(f"<div class='ai-reply'>{response_text}</div>", unsafe_allow_html=True)
                except Exception as e:
                    logger.error(f"Error in stream_response: {str(e)}")
                    response_placeholder.error(f"❌ Error: {str(e)}")
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
                loop.run_until_complete(stream_response())
            except Exception as e:
                logger.error(f"Error with event loop: {str(e)}")
                st.error("An error occurred while processing your request.")
    else:
        st.info("Upload or capture a photo to begin analysis.")

if __name__ == "__main__":
    main()
