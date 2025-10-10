# 🍽️ FitPlate AI

A smart food recognition and nutrition tracking app that uses AI to analyze meals and provide personalized dietary advice.

## 🌟 Features

- **Food Recognition**: Upload or take photos of meals for instant identification
- **Nutrition Analysis**: Get detailed macro breakdown (calories, protein, carbs, fat)
- **Smart Suggestions**: Receive personalized dietary advice based on your goals
- **Chat Interface**: Ask questions about your meals and get AI-powered answers
- **Profile Support**: Different modes for bulking, cutting, or maintenance

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fitplate-ai.git
cd fitplate-ai
```

2. Set up your environment:
```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

3. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

4. Run the app:
```bash
poetry run streamlit run app/main.py
```

## 🔧 Project Structure

```
fitplate-ai/
├── app/
│   └── main.py           # Streamlit UI
├── services/
│   ├── image_recognition.py   # Image analysis using OpenAI
│   ├── nutrition_lookup.py    # Nutrition data lookup
│   └── suggestions.py         # Personalized recommendations
├── data/
│   ├── meals.json            # Nutrition database
│   └── user_profiles.json    # User preferences
├── models/                   # For future ML models
├── utils/                   # Helper functions
├── pyproject.toml          # Poetry dependencies
├── Dockerfile             # Container definition
└── README.md
```

## 🐋 Docker Support

Build and run with Docker:

```bash
# Build the image
docker build -t fitplate-ai .

# Run the container
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key_here fitplate-ai
```

## 🛠️ Configuration

The app can be configured through environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `NUTRITIONIX_APP_ID`: Optional Nutritionix API credentials
- `NUTRITIONIX_API_KEY`: Optional Nutritionix API credentials

## 📝 Usage Examples

1. **Analyze a Meal**:
   - Upload a photo or use your camera
   - Click "Analyze Dish"
   - View nutrition breakdown and suggestions

2. **Chat About Your Meal**:
   - After analysis, use the chat interface
   - Ask questions like:
     - "Is this good for bulking?"
     - "How can I make this healthier?"
     - "What's a good side dish?"

3. **Customize Your Profile**:
   - Select a profile type (bulking/cutting/maintenance)
   - Get personalized suggestions
   - Track your meal history

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -am 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for their powerful Vision and Chat APIs
- Streamlit for the awesome web framework
- The Python community for various dependencies
