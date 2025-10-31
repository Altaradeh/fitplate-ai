"""
Authentication page for user login/registration.
"""
import streamlit as st
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.auth import AuthService, login_user, logout_user, is_authenticated
from app.database import init_database

logger = logging.getLogger(__name__)

def show_auth_page():
    """Display authentication page"""
    st.set_page_config(page_title="🍽️ FitPlate AI - Login", page_icon="🍽️", layout="centered")
    
    # Initialize database
    try:
        init_database()
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        return
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1>🍽️ FitPlate AI</h1>
        <p style="font-size: 1.2rem; color: #666;">Track your nutrition journey with AI-powered food analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if already authenticated
    if is_authenticated():
        st.success(f"Welcome back, {st.session_state.username}!")
        if st.button("Continue to App"):
            st.switch_page("main.py")
        if st.button("Logout"):
            logout_user()
            st.rerun()
        return
    
    # Authentication tabs
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        show_login_form()
    
    with tab2:
        show_registration_form()

def show_login_form():
    """Display login form"""
    st.markdown("### Welcome Back!")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login", use_container_width=True)
        
        if submit:
            if not username or not password:
                st.error("Please enter both username and password")
                return
            
            user = AuthService.authenticate_user(username, password)
            if user:
                login_user(user)
                st.success("Login successful!")
                st.balloons()
                # Store user_id and username in localStorage for persistent login
                st.markdown(f"""
                <script>
                window.localStorage.setItem('fitplate_user_id', '{user.id}');
                window.localStorage.setItem('fitplate_username', '{user.username}');
                </script>
                """, unsafe_allow_html=True)
                # Redirect to main app
                st.switch_page("main.py")
            else:
                st.error("Invalid username or password")

def show_registration_form():
    """Display registration form"""
    st.markdown("### Create Your Account")
    
    with st.form("registration_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("Username*")
            email = st.text_input("Email*")
            password = st.text_input("Password*", type="password")
            confirm_password = st.text_input("Confirm Password*", type="password")
        
        with col2:
            # Initial preferences
            st.markdown("**Initial Preferences**")
            goal = st.selectbox("Fitness Goal", ["Balanced", "Bulking", "Cutting", "Maintenance"])
            diet = st.selectbox("Diet Type", ["None", "Keto", "Vegan", "Vegetarian", "Mediterranean"])
            calorie_target = st.number_input("Daily Calorie Target", min_value=1200, max_value=5000, value=2500, step=100)
        
        # Optional personal info
        with st.expander("Personal Information (Optional)"):
            col3, col4 = st.columns(2)
            with col3:
                age = st.number_input("Age", min_value=13, max_value=120, value=None)
                gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
            with col4:
                height = st.number_input("Height (cm)", min_value=100, max_value=250, value=None)
                weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=None)
            
            activity_level = st.selectbox("Activity Level", 
                                        ["Sedentary", "Light", "Moderate", "Active", "Very Active"],
                                        index=2)
        
        health_conditions = st.multiselect("Health Conditions", 
                                         ["Diabetes", "Heart disease", "High blood pressure", 
                                          "High cholesterol", "Celiac Disease", "Lactose Intolerance"])
        
        submit = st.form_submit_button("Create Account", use_container_width=True)
        
        if submit:
            # Validation
            if not all([username, email, password, confirm_password]):
                st.error("Please fill in all required fields")
                return
            
            if password != confirm_password:
                st.error("Passwords do not match")
                return
            
            if len(password) < 6:
                st.error("Password must be at least 6 characters long")
                return
            
            # Create user with initial preferences
            initial_prefs = {
                'goal': goal,
                'diet': diet,
                'calorie_target': calorie_target,
                'health_conditions': health_conditions,
                'age': age if age else None,
                'gender': gender if gender else None,
                'height_cm': height if height else None,
                'weight_kg': weight if weight else None,
                'activity_level': activity_level
            }
            
            user = AuthService.create_user(username, email, password, initial_prefs)
            if user:
                st.success("Account created successfully!")
                st.info("Please login with your new account")
            else:
                st.error("Failed to create account. Username or email may already exist.")

if __name__ == "__main__":
    show_auth_page()