import asyncio
import streamlit as st
from PIL import Image
from openai import AsyncOpenAI
import os
import nest_asyncio
nest_asyncio.apply()
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
os.environ["OPENAI_API_KEY"] = os.getenv("openai_key")
client = AsyncOpenAI()

async def analyze_dish(image):
    await asyncio.sleep(0.5)  # simulate network call
    return {"dish_name": "Grilled Chicken Salad", "confidence": 0.91}

async def estimate_nutrition(dish_name):
    await asyncio.sleep(0.4)
    return {"Calories": 420, "Protein (g)": 38, "Carbs (g)": 18, "Fat (g)": 14}

def suggest_improvements(nutrition):
    tips = []
    if nutrition["Protein (g)"] < 30:
        tips.append("Add more protein (e.g., eggs, beans, chicken).")
    if nutrition["Carbs (g)"] > 60:
        tips.append("Reduce carbs (add more vegetables).")
    if nutrition["Fat (g)"] > 25:
        tips.append("Use leaner oils or lighter cooking methods.")
    return tips or ["Perfect balance! Keep it up 💪"]

async def answer_question_stream(question, dish_info, nutrition):
    nutrition_text = ", ".join([f"{k}: {v}" for k, v in nutrition.items()])
    dish_name = dish_info["dish_name"]
    messages = [
        {"role": "system", "content": "You are a helpful nutrition and fitness assistant. Answer using evidence-based guidance."},
        {"role": "user", "content": f"The meal is {dish_name}. Nutrition: {nutrition_text}. Question: {question}"}
    ]
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
        max_tokens=250
    )
    collected = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        collected += delta
        yield delta
         # End of async generator; do not return a value

async def process_image(image):
    dish = await analyze_dish(image)
    nutrition = await estimate_nutrition(dish["dish_name"])
    return dish, nutrition

def main():
    st.set_page_config(page_title="🍽️ Smart Dish Analyzer (Async + Streaming)")
    st.title("🍽️ Smart Dish Analyzer")
    st.write("Upload a meal photo or use your camera to analyze your food and chat about your goals.")

    tab_upload, tab_camera = st.tabs(["📂 Upload Photo", "📷 Take Photo"])
    image = None
    with tab_upload:
        uploaded_file = st.file_uploader("Upload a dish photo", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Dish", use_container_width=True)
    with tab_camera:
        camera_input = st.camera_input("Take a photo with your camera")
        if camera_input:
            image = Image.open(camera_input)
            st.image(image, caption="Captured Dish", use_container_width=True)

    if image:
        if st.button("Analyze Dish"):
            loop = asyncio.get_event_loop()
            dish, nutrition = loop.run_until_complete(process_image(image))
            st.success(f"Dish: {dish['dish_name']} ({dish['confidence']*100:.1f}% confidence)")
            st.table(nutrition)
            for tip in suggest_improvements(nutrition):
                st.write("- " + tip)

            st.markdown("### 💬 Ask About Your Meal")
            user_question = st.chat_input("Ask something like 'Is this good for bulking?'")
            if user_question:
                chat_placeholder = st.empty()
                chat_placeholder.chat_message("user").write(user_question)
                chat_box = st.chat_message("assistant")
                response_text = ""
                async def stream_response():
                    try:
                        found_response = False
                        async for token in answer_question_stream(user_question, dish, nutrition):
                            found_response = True
                            response_text += token
                            chat_box.markdown(response_text)
                        if not found_response:
                            chat_box.markdown(":warning: No response received from OpenAI. Check your API key, model, or network.")
                    except Exception as e:
                        chat_box.markdown(f":red[Error: {e}]")
                loop.run_until_complete(stream_response())
    else:
        st.info("Upload or capture a photo to begin analysis.")

if __name__ == "__main__":
    main()
