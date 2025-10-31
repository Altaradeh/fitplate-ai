"""
Food tracking and history page.
"""
import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime, timedelta, timezone

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.auth import is_authenticated, get_current_user_id, get_current_username
from app.meal_tracking import MealTrackingService
import logging

logger = logging.getLogger(__name__)

def show_tracking_page():
    """Display food tracking page"""
    st.set_page_config(page_title="🍽️ Food Tracking", page_icon="🍽️", layout="wide")
    
    # Check authentication
    if not is_authenticated():
        st.error("Please login to access food tracking")
        if st.button("Go to Login"):
            st.switch_page("pages/1_Login.py")
        return
    
    user_id = get_current_user_id()
    username = get_current_username()
    
    if not user_id:
        st.error("Invalid user session. Please login again.")
        return
    
    st.title(f"📊 Food Tracking Dashboard - {username}")
    
    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["� Dashboard", "🍽️ Recent Meals", "📅 Meal History"])
    
    with tab1:
        show_dashboard(user_id)
    
    with tab2:
        show_recent_meals(user_id)
    
    with tab3:
        show_meal_history(user_id)

def show_dashboard(user_id: int):
    """Show nutrition dashboard"""
    st.header("Today's Nutrition Dashboard")
    
    # Get today's summary
    today_summary = MealTrackingService.get_daily_nutrition_summary(user_id)
    
    if not today_summary:
        st.info("No meals logged today. Start by analyzing a meal!")
        return
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Calories",
            value=f"{today_summary['total_calories']:.0f}",
            help="Total calories consumed today"
        )
    
    with col2:
        st.metric(
            label="Protein",
            value=f"{today_summary['total_protein']:.1f}g",
            help="Total protein consumed today"
        )
    
    with col3:
        st.metric(
            label="Carbs",
            value=f"{today_summary['total_carbs']:.1f}g",
            help="Total carbohydrates consumed today"
        )
    
    with col4:
        st.metric(
            label="Fat",
            value=f"{today_summary['total_fat']:.1f}g",
            help="Total fat consumed today"
        )
    
    # Today's meals
    if today_summary['meals']:
        st.subheader("Today's Meals")
        
        for meal in today_summary['meals']:
            with st.expander(f"{meal['type'].title()}: {meal['name']} ({meal['calories']:.0f} kcal)", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Time:** {meal['time'].strftime('%H:%M')}")
                    st.write(f"**Calories:** {meal['calories']:.0f} kcal")
                    st.write(f"**Protein:** {meal['protein']:.1f}g")
                    st.write(f"**Carbs:** {meal['carbs']:.1f}g")
                    st.write(f"**Fat:** {meal['fat']:.1f}g")
                    
                    if meal['health_score']:
                        st.write(f"**Health Score:** {meal['health_score']}/100")
                    
                    if meal['rating']:
                        st.write(f"**Your Rating:** {'⭐' * meal['rating']}")
                
                with col2:
                    if st.button(f"Delete", key=f"delete_{meal['id']}"):
                        if MealTrackingService.delete_meal(meal['id'], user_id):
                            st.success("Meal deleted!")
                            st.rerun()
                        else:
                            st.error("Failed to delete meal")

def show_recent_meals(user_id: int):
    """Show recent meals"""
    st.header("Recent Meals (Last 7 Days)")
    
    # Filter options
    col1, col2 = st.columns([1, 1])
    with col1:
        days = st.selectbox("Time Period", [7, 14, 30], index=0, format_func=lambda x: f"Last {x} days")
    with col2:
        meal_type_filter = st.selectbox("Meal Type", ["All", "breakfast", "lunch", "dinner", "snack"])
    
    # Get recent meals
    recent_meals = MealTrackingService.get_recent_meals(user_id, days=days, confirmed_only=True)
    
    if not recent_meals:
        st.info(f"No confirmed meals found in the last {days} days.")
        return
    
    # Filter by meal type
    if meal_type_filter != "All":
        recent_meals = [meal for meal in recent_meals if meal.meal_type == meal_type_filter]  # type: ignore
    
    # Create DataFrame for display
    meal_data = []
    for meal in recent_meals:
        nutrition = meal.nutrition_data.get('nutrition_summary', {})
        calories = nutrition.get('calories', {}).get('value', 0)
        macros = nutrition.get('macros', {})
        
        # Ensure health_score is an integer or None
        hs = getattr(meal, 'health_score', None)
        try:
            hs = int(hs) if hs is not None and not isinstance(hs, str) else None
        except Exception:
            hs = None
        meal_data.append({
            'Date': meal.consumed_at.strftime('%Y-%m-%d'),
            'Time': meal.consumed_at.strftime('%H:%M'),
            'Meal': meal.meal_name,
            'Type': meal.meal_type.title(),
            'Calories': calories,
            'Protein (g)': macros.get('protein', {}).get('value', 0),
            'Carbs (g)': macros.get('carbs', {}).get('value', 0),
            'Fat (g)': macros.get('fat', {}).get('value', 0),
            'Health Score': hs,
            # Rating removed from UI
        })
    if meal_data:
        df = pd.DataFrame(meal_data)
        st.dataframe(df, use_container_width=True)
        
        # Summary stats
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col3:
            # Only use numeric health scores for averaging
            if 'Health Score' in df.columns:
                numeric_scores = pd.to_numeric(df['Health Score'], errors='coerce')
                valid_scores = numeric_scores.dropna()
                avg_health_score = valid_scores.mean() if not valid_scores.empty else None
            else:
                avg_health_score = None
            st.metric("Average Health Score", f"{avg_health_score:.0f}/100" if avg_health_score is not None else "Not analyzed")
        
        with col2:
            total_meals = len(df)
            st.metric("Total Meals Logged", total_meals)

def show_meal_history(user_id: int):
    """Show meal history with search and filters"""
    st.header("Meal History")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now().date())
    
    # Search and filters
    search_term = st.text_input("Search meals", placeholder="Enter meal name to search...")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        meal_type_filter = st.selectbox("Meal Type", ["All", "breakfast", "lunch", "dinner", "snack"], key="history_meal_type")
    with col2:
        min_health_score = st.slider("Minimum Health Score", 0, 100, 0)
    with col3:
        show_unconfirmed = st.checkbox("Include unconfirmed meals")
    
    # Get meals in date range
    start_datetime = datetime.combine(start_date, datetime.min.time().replace(tzinfo=timezone.utc))
    end_datetime = datetime.combine(end_date, datetime.max.time().replace(tzinfo=timezone.utc))
    
    # This would require modifying MealTrackingService to support date range queries
    # For now, get recent meals and filter
    all_meals = MealTrackingService.get_recent_meals(user_id, days=365, confirmed_only=not show_unconfirmed)
    
    # Filter meals
    filtered_meals = []
    for meal in all_meals:
        # Date filter
        meal_dt = meal.consumed_at
        # Only filter if meal_dt is a datetime object
        import datetime as _dt
        if not isinstance(meal_dt, _dt.datetime):
            continue
        # Ensure meal_dt is timezone-aware (UTC)
        if meal_dt.tzinfo is None or meal_dt.tzinfo.utcoffset(meal_dt) is None:
            meal_dt = meal_dt.replace(tzinfo=_dt.timezone.utc)
        if not (start_datetime <= meal_dt <= end_datetime):
            continue
        
        # Meal type filter
        if meal_type_filter != "All" and meal.meal_type != meal_type_filter:  # type: ignore
            continue
        
        # Health score filter
        if meal.health_score and meal.health_score < min_health_score:  # type: ignore
            continue
        
        # Search filter
        if search_term and search_term.lower() not in meal.meal_name.lower():
            continue
        
        filtered_meals.append(meal)
    
    if not filtered_meals:
        st.info("No meals found matching your criteria.")
        return
    
    # Display results
    st.write(f"Found {len(filtered_meals)} meals")
    
    # Group by date
    meals_by_date = {}
    for meal in filtered_meals:
        date_key = meal.consumed_at.date()
        if date_key not in meals_by_date:
            meals_by_date[date_key] = []
        meals_by_date[date_key].append(meal)
    
    # Display by date
    for date, day_meals in sorted(meals_by_date.items(), reverse=True):
        with st.expander(f"📅 {date.strftime('%A, %B %d, %Y')} ({len(day_meals)} meals)", expanded=False):
            for meal in sorted(day_meals, key=lambda x: x.consumed_at):
                nutrition = meal.nutrition_data.get('nutrition_summary', {})
                calories = nutrition.get('calories', {}).get('value', 0)
                
                st.markdown(f"""
                **{meal.consumed_at.strftime('%H:%M')} - {meal.meal_type.title()}:** {meal.meal_name}
                - Calories: {calories:.0f} kcal
                - Health Score: {(int(meal.health_score) if meal.health_score and meal.health_score > 0 else 'Not analyzed')}/100
                - Confirmed: {'✅' if meal.confirmed else '❌'}
                
                """)
                
                if meal.notes:
                    st.write(f"*Notes: {meal.notes}*")

if __name__ == "__main__":
    show_tracking_page()