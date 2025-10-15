# 🍽️ FitPlate AI

A smart food recognition and nutrition tracking app that uses AI to analyze meals and provide personalized dietary advice.

## 🌟 Features

### 🔍 **Smart Food Analysis**
- **Food Recognition**: Upload or take photos of meals for instant identification
- **Non-Food Detection**: Robust error handling when non-food items are uploaded
- **Nutrition Analysis**: Get detailed macro breakdown (calories, protein, carbs, fat)
- **Component Analysis**: Break down complex meals into individual components

### 🎯 **Personalized Experience**
- **User Preferences**: Comprehensive profile system with goals, diet types, and health conditions
- **Multi-Select Health Conditions**: Support for multiple dietary restrictions and health needs
- **Dynamic Chat Suggestions**: AI-generated conversation starters based on your profile
- **Personalized AI Responses**: All recommendations tailored to your specific goals and restrictions

### 💬 **Interactive Chat System**
- **Contextual Q&A**: Ask questions about your meals and get AI-powered answers
- **Smart Suggestions**: Personalized question suggestions based on your dietary profile
- **Real-time Responses**: Streaming chat responses for immediate feedback

### 🎨 **Modern UI/UX**
- **Centralized Icon System**: Consistent visual language with 130+ food and nutrition icons
- **Responsive Design**: Clean, modern interface that works on all devices
- **Demo Mode**: Test the app without API keys using sample data
- **Dark/Light Theme**: Automatic theme detection with manual toggle

### 🏥 **Health & Diet Support**
- **Diet Types**: Keto, Vegan, Vegetarian, Mediterranean, and more
- **Health Conditions**: Diabetes, Heart Disease, Celiac Disease, Lactose Intolerance, etc.
- **Goal-Based Recommendations**: Bulking, Cutting, Maintenance, or Balanced nutrition
- **Calorie Tracking**: Custom daily calorie targets with progress tracking

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
│   ├── main.py               # Streamlit UI with personalized features
│   ├── models.py             # Pydantic data models with validation
│   ├── config.py             # App configuration and settings
│   ├── styles.css            # Custom CSS styling
│   └── icons.json            # Centralized icon system (130+ icons)
├── services/
│   ├── analyzer.py           # Core food analysis service with personalization
│   ├── image_recognition.py  # OpenAI Vision API integration
│   ├── nutrition_lookup.py   # Nutrition data processing
│   └── suggestions.py        # AI-powered recommendations
├── data/
│   ├── meals.json            # Sample meal data
│   ├── demo_meal.json        # Demo mode data
│   └── user_profiles.json    # User preference templates
├── models/                   # ML model storage (future use)
├── utils/                    # Helper utilities
│   └── storage.py            # Data persistence utilities
├── tests/                    # Test suite
│   ├── test_nutrition.py     # Nutrition analysis tests
│   └── __init__.py
├── pyproject.toml            # Poetry dependencies and config
├── poetry.lock               # Locked dependency versions
├── .env                      # Environment variables (create from .env.example)
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

The app supports multiple configuration options:

### **Environment Variables**
- `OPENAI_API_KEY`: Your OpenAI API key (required for AI features)
- `UX_DEMO`: Set to `true` to enable demo mode (no API required)
- `UX_DATA_JSON`: Path to demo data file (default: `data/demo_meal.json`)

### **User Preferences** (Saved in browser localStorage)
- **Goals**: Balanced, Bulking, Cutting, Maintenance
- **Diet Types**: None, Keto, Vegan, Vegetarian, Mediterranean
- **Health Conditions**: Multiple selection support for Diabetes, Heart Disease, etc.
- **Calorie Targets**: Custom daily calorie goals

### **Demo Mode**
Test all features without API keys:
```bash
# Enable demo mode
export UX_DEMO=true
poetry run streamlit run app/main.py
```

## 📝 Usage Examples

### 1. **Analyze a Meal**
- Upload a photo from your device or take a new photo
- The AI automatically detects food items and nutrition information
- View detailed breakdown including calories, macros, and health warnings
- Get personalized suggestions based on your dietary profile

### 2. **Personalized Chat Experience**
After analysis, use the intelligent chat system:
- **Goal-Based Questions**: "Does this meal provide enough protein for bulking?"
- **Diet-Specific Queries**: "Is this meal suitable for a Keto diet?"
- **Health-Conscious Questions**: "Is this safe for my diabetes?"
- **Improvement Suggestions**: "How can I make this meal healthier?"

### 3. **Customize Your Profile**
Set up your personalized nutrition profile:
- **Choose Your Goal**: Bulking (high protein/calories), Cutting (calorie deficit), Maintenance, or Balanced
- **Select Diet Type**: Keto, Vegan, Vegetarian, Mediterranean, or None
- **Add Health Conditions**: Multiple selections for Diabetes, Heart Disease, Celiac, etc.
- **Set Calorie Target**: Custom daily calorie goals for tracking

### 4. **Multi-Select Health Conditions**
Unlike basic apps, FitPlate AI supports multiple health conditions:
```
✅ Diabetes + Heart Disease
✅ Celiac Disease + Lactose Intolerance  
✅ High Blood Pressure + High Cholesterol
```

### 5. **Smart Error Handling**
- Upload non-food images (faces, objects) and get helpful guidance
- Robust validation prevents crashes from unexpected inputs
- Clear feedback when meals can't be analyzed

### 6. **Demo Mode Testing**
Try the app without API setup:
```bash
# Enable demo mode in the sidebar
# Or set environment variable
export UX_DEMO=true
```

## 🏗️ Technical Architecture

### **AI Integration**
- **OpenAI GPT-4o Vision**: Advanced food recognition and analysis
- **OpenAI GPT-4o Chat**: Contextual nutrition advice and Q&A
- **Streaming Responses**: Real-time chat with immediate feedback
- **Personalized Prompts**: AI context includes user preferences for relevant advice

### **Data Validation & Error Handling**
- **Pydantic Models**: Type-safe data validation with automatic serialization
- **Non-Food Detection**: Graceful handling of invalid image uploads
- **Robust Error Recovery**: User-friendly error messages and fallback options
- **Input Sanitization**: Safe handling of user inputs and API responses

### **Modern Frontend**
- **Streamlit Framework**: Clean, responsive web interface
- **Component Caching**: Efficient image analysis with result caching
- **Local Storage**: Browser-based preference persistence
- **Progressive Enhancement**: Works with or without JavaScript

### **Icon System**
- **Centralized Management**: 130+ icons in `app/icons.json`
- **Smart Matching**: Automatic icon selection based on content
- **Fallback System**: Graceful degradation for unknown terms
- **Easy Extension**: Add new icons without code changes

## 🔄 Recent Updates

### **v2.0 - Personalization & UX Overhaul**
- ✅ **Multi-select health conditions** replacing single-select dropdowns
- ✅ **Personalized chat suggestions** based on user profile
- ✅ **Centralized icon system** with 130+ food and nutrition icons
- ✅ **Enhanced error handling** for non-food image uploads
- ✅ **User preferences integration** throughout AI analysis pipeline
- ✅ **Improved UI/UX** with modern card-based design
- ✅ **Demo mode** for testing without API keys

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**:
   - Follow the existing code style
   - Add icons to `app/icons.json` for new food items
   - Update tests if needed
4. **Test your changes**:
   ```bash
   poetry run python -m pytest tests/
   ```
5. **Commit and push**:
   ```bash
   git commit -am 'Add amazing feature'
   git push origin feature/amazing-feature
   ```
6. **Open a Pull Request**

### **Contributing Guidelines**
- **Icons**: Add new food/nutrition icons to `app/icons.json`
- **Models**: Update Pydantic models in `app/models.py` for new data types
- **Personalization**: Ensure new features use `user_preferences` parameter
- **Error Handling**: Include graceful fallbacks for API failures
- **Documentation**: Update README for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** for their powerful GPT-4o Vision and Chat APIs
- **Streamlit** for the intuitive web framework
- **Pydantic** for robust data validation and serialization
- **Poetry** for modern Python dependency management
- **The Python Community** for the excellent ecosystem of tools and libraries

---

**Built with ❤️ for better nutrition tracking and healthier eating habits**
