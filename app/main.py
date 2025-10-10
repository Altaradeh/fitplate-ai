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
    """Load custom CSS styles."""
    st.markdown("""
        <style>
        .meal-card {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 0.75rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin-bottom: 1rem;
        }
        
        .metric-grid {
            display: grid;
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .nutrient-tag {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            margin: 0.25rem;
            border-radius: 1rem;
            background-color: #f3f4f6;
            color: #374151;
            font-size: 0.875rem;
            font-weight: 500;
        }
        
        .section-title {
            color: #111827;
            font-size: 1.25rem;
            font-weight: 600;
            margin: 1.5rem 0 1rem;
        }
        
        .insight-card {
            background-color: #f8fafc;
            padding: 1rem;
            border-left: 4px solid #3b82f6;
            margin: 0.5rem 0;
        }
        
        .warning-tag {
            color: #dc2626;
            background-color: #fee2e2;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.875rem;
        }
        </style>
    """, unsafe_allow_html=True)

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

def main():
    """Main application entry point."""
    # Configure the page
    st.set_page_config(
        page_title="🍽️ Smart Dish Analyzer",
        page_icon="🍽️",
        layout="centered"
    )
    
    # Load custom styles
    load_styles()
    
    # Initialize the analyzer service
    analyzer = FoodAnalyzerService(api_key)
    
    st.title("🍽️ Smart Dish Analyzer")
    
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
        # Custom styling to hide filename
        st.markdown(
            '''
            <style>
            /* Hide the filename display */
            .stFileUploader > div:first-child ~ div:not(:last-child) {display: none;}
            /* Hide "Drag and drop file here" text */
            .stFileUploader > div:first-child > div:first-child {display: none;}
            /* Keep the upload button visible */
            .stFileUploader > div:last-child {display: block !important;}
            </style>
            ''',
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader("Upload a dish photo", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
    with tab_camera:
        camera_input = st.camera_input("Take a photo")
        if camera_input:
            image = Image.open(camera_input)

    if image:
        # Display the image in a centered layout
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Your Dish", use_container_width=True)
            
        with st.spinner("Analyzing your dish..."):
            # Analyze the image and get nutrition data
            loop = asyncio.get_event_loop()
            dish, nutrition = loop.run_until_complete(process_image(image))
            
            # Create analysis container
            analysis_container = st.container()
            with analysis_container:
                st.markdown('<p class="section-header">📊 Meal Analysis</p>', unsafe_allow_html=True)
                
                # Dish identification
                confidence = dish['confidence'] * 100
                confidence_emoji = "🎯" if confidence >= 90 else "✨" if confidence >= 70 else "👀"
                st.success(f"{confidence_emoji} **{dish['dish_name']}** ({confidence:.1f}% confidence)")
                
                # Main nutrition card
                main_card = st.container()
                with main_card:
                    # Top row: Calories and tags
                    cal_col, tags_col = st.columns([1, 2])
                    with cal_col:
                        if not nutrition or not nutrition.get("nutrition_summary"):
                            st.error("⚠️ Could not analyze nutrition information")
                            return
                        
                        # Extract nutrition data from the new structure
                        nutrition_summary = nutrition["nutrition_summary"]
                        cal_dv = nutrition_summary["calories"]["daily_value"]
                        st.metric(
                            "Total Calories",
                            f"{nutrition_summary['calories']['value']} kcal",
                            f"{cal_dv}% Daily Value",
                            help="Based on 2000 kcal daily requirement",
                            delta_color="off"
                        )
                        meal_info = nutrition.get("meal_info", {})
                        serving_size = meal_info.get("serving_size", "N/A")
                        st.caption(f"📏 Serving: {serving_size}")
                    
                    with tags_col:
                        diet_tags = nutrition_summary.get("diet_tags", [])
                        if diet_tags:
                            tags_html = " ".join([
                                f'<span class="diet-tag">🏷️ {tag}</span>'
                                for tag in diet_tags
                            ])
                            st.markdown(f"<div style='margin-top: 1rem;'>{tags_html}</div>", unsafe_allow_html=True)
                
                # Macronutrients grid
                st.markdown('<p class="subsection-header">Macronutrients</p>', unsafe_allow_html=True)
                macro_cols = st.columns(3)
                
                macros = nutrition_summary["macros"]
                
                with macro_cols[0]:
                    st.metric(
                        "🥩 Protein",
                        f"{macros['protein']['value']:.1f}g",
                        f"{macros['protein']['daily_value']}% DV",
                        help="Based on 50g daily protein requirement",
                        delta_color="off"
                    )
                
                with macro_cols[1]:
                    st.metric(
                        "🍚 Carbohydrates",
                        f"{macros['carbs']['value']:.1f}g",
                        f"{macros['carbs']['daily_value']}% DV",
                        help="Based on 225g daily carbohydrate requirement",
                        delta_color="off"
                    )
                
                with macro_cols[2]:
                    st.metric(
                        "🫒 Fat",
                        f"{macros['fat']['value']:.1f}g",
                        f"{macros['fat']['daily_value']}% DV",
                        help="Based on 65g daily fat requirement",
                        delta_color="off"
                    )
                
                # Additional nutrients
                st.markdown('<p class="subsection-header">Additional Nutrients</p>', unsafe_allow_html=True)
                detail_cols = st.columns(2)
                
                additional = nutrition_summary["additional"]
                
                with detail_cols[0]:
                    st.metric(
                        "🌾 Fiber",
                        f"{additional['fiber']['value']:.1f}g",
                        f"{additional['fiber']['daily_value']}% DV",
                        help="Based on 28g daily fiber requirement",
                        delta_color="off"
                    )
                    
                    st.metric(
                        "🍯 Sugar",
                        f"{additional['sugar']['value']:.1f}g",
                        "No DV established",
                        help="Includes both natural and added sugars",
                        delta_color="off"
                    )
                
                with detail_cols[1]:
                    st.metric(
                        "🥑 Saturated Fat",
                        f"{additional['saturated_fat']['value']:.1f}g",
                        f"{additional['saturated_fat']['daily_value']}% DV",
                        help="Based on 20g daily saturated fat limit",
                        delta_color="off"
                    )
                
                # Item Breakdown
                st.markdown('<p class="section-header">🍽️ Meal Components</p>', unsafe_allow_html=True)
                
                # Display each item in a modern card layout
                components = nutrition.get("components", [])
                for item in components:
                    with st.container():
                        # Item header with type icon
                        type_icons = {
                            "main_dish": "🍽️",
                            "side_dish": "🥗",
                            "beverage": "🥤",
                            "condiment": "🧂"
                        }
                        icon = type_icons.get(item['type'], "🍴")
                        
                        # Item name and type
                        st.markdown(f"### {icon} {item['name']}")
                        st.caption(f"Type: {item['type'].replace('_', ' ').title()} | {item['serving']}")
                        
                        # Create three columns for the main nutrients
                        c1, c2, c3 = st.columns(3)
                        
                        item_nutrition = item['nutrition']
                        
                        with c1:
                            st.metric(
                                "Calories",
                                f"{item_nutrition['calories']} kcal",
                                delta=None
                            )
                        
                        with c2:
                            st.metric(
                                "Protein",
                                f"{item_nutrition['protein']:.1f}g",
                                delta=None
                            )
                        
                        with c3:
                            st.metric(
                                "Carbs",
                                f"{item_nutrition['carbs']:.1f}g",
                                delta=None
                            )
                        
                        # Additional details in expandable section
                        with st.expander("More Details"):
                            det1, det2 = st.columns(2)
                            with det1:
                                st.metric("Fat", f"{item_nutrition['fat']:.1f}g")
                                st.metric("Fiber", f"{item_nutrition['fiber']:.1f}g")
                            with det2:
                                st.metric("Sugar", f"{item_nutrition['sugar']:.1f}g")
                                st.metric("Saturated Fat", f"{item_nutrition['saturated_fat']:.1f}g")
                            
                            if item.get('diet_tags'):
                                tags_html = " ".join([
                                    f'<span class="diet-tag">{tag}</span>'
                                    for tag in item['diet_tags']
                                ])
                                st.markdown(f"**Diet Tags:** <div>{tags_html}</div>", unsafe_allow_html=True)
                        
                        st.markdown("---")
                
                # Warnings and Suggestions
                suggestions_col1, suggestions_col2 = st.columns(2)
                
                with suggestions_col1:
                    warnings = nutrition_summary.get("warnings", [])
                    if warnings:
                        with st.expander("⚠️ Warnings", expanded=False):
                            for warning in warnings:
                                st.warning(warning, icon="⚠️")

                with suggestions_col2:
                    with st.expander("💡 AI Suggestions", expanded=False):
                        with st.spinner("Getting personalized suggestions..."):
                            tips = loop.run_until_complete(analyzer.suggest_improvements(nutrition))
                            for tip in tips:
                                st.write(tip)

                # Chat section
                st.markdown("---")
                user_question = st.chat_input("💬 Ask about your meal (e.g., 'Is this good for bulking?')")
                if user_question:
                    # Compact chat UI
                    with st.container():
                        st.chat_message("user", avatar="🤔").write(user_question)
                        chat_box = st.chat_message("assistant", avatar="🍽️")
                        response_placeholder = chat_box.empty()
                        
                    async def stream_response():
                        response_text = ""
                        try:
                            async for token in handle_chat(analyzer, user_question, nutrition):
                                response_text += token
                                response_placeholder.markdown(response_text)
                        except Exception as e:
                            logger.error(f"Error in stream_response: {str(e)}")
                            response_placeholder.error(f"❌ Error: {str(e)}")
                    
                    # Run the streaming response
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_closed():
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        loop.run_until_complete(stream_response())
                    except Exception as e:
                        logger.error(f"Error with event loop: {str(e)}")
                        st.error("An error occurred while processing your request.")
    else:
        st.info("Upload or capture a photo to begin analysis.")

if __name__ == "__main__":
    main()
