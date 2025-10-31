"""
Nutrition reports and analytics page.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from datetime import datetime, timedelta, timezone

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.auth import is_authenticated, get_current_user_id, get_current_username
from app.reporting import ReportingService
import logging

logger = logging.getLogger(__name__)

def show_reports_page():
    """Display reports and analytics page"""
    st.set_page_config(page_title="📈 Nutrition Reports", page_icon="📈", layout="wide")
    
    # Check authentication
    if not is_authenticated():
        st.error("Please login to access reports")
        if st.button("Go to Login"):
            # Use correct relative path for Streamlit pages
            st.switch_page("pages/1_Login.py")
        return
    
    user_id = get_current_user_id()
    username = get_current_username()
    
    if not user_id:
        st.error("Invalid user session. Please login again.")
        return
    
    st.title(f"📈 Nutrition Reports - {username}")
    
    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["📊 Weekly Report", "📅 Monthly Report", "📈 Progress Tracking"])
    
    with tab1:
        show_weekly_report(user_id)
    
    with tab2:
        show_monthly_report(user_id)
    
    with tab3:
        show_progress_tracking(user_id)

def show_weekly_report(user_id: int):
    """Show weekly nutrition report"""
    st.header("Weekly Nutrition Report")
    
    # Week selector
    col1, col2 = st.columns([1, 2])
    with col1:
        weeks_back = st.selectbox("Select Week", 
                                 options=list(range(8)), 
                                 format_func=lambda x: "This week" if x == 0 else f"{x} week{'s' if x > 1 else ''} ago")
    
    # Calculate start date
    today = datetime.now(timezone.utc).date()
    start_date_dt = today - timedelta(days=today.weekday() + (weeks_back * 7))
    start_datetime = datetime.combine(start_date_dt, datetime.min.time()).replace(tzinfo=timezone.utc)
    
    # Generate report
    with st.spinner("Generating weekly report..."):
        report = ReportingService.generate_weekly_report(user_id, start_datetime)
    
    if not report:
        st.warning("No data available for the selected week.")
        return
    
    # Display report
    # Convert period string to datetime before calling .date()
    
    st.subheader(f"Report for {report['period']}")
    
    # Summary metrics
    summary = report['summary']
    targets = report['targets']
    adherence = report['adherence']
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_cal = summary['avg_calories'] - targets['calories']
        st.metric(
            "Avg Daily Calories",
            f"{summary['avg_calories']:.0f}",
            f"{delta_cal:+.0f} vs target",
            delta_color="normal"
        )
    
    with col2:
        delta_protein = summary['avg_protein'] - targets['protein']
        st.metric(
            "Avg Daily Protein",
            f"{summary['avg_protein']:.1f}g",
            f"{delta_protein:+.1f}g vs target"
        )
    
    with col3:
        st.metric(
            "Total Meals Logged",
            summary['total_meals'],
            f"{summary['avg_meals_per_day']:.1f} per day"
        )
    
    with col4:
        avg_adherence = sum(adherence.values()) / len(adherence)
        st.metric(
            "Goal Adherence",
            f"{avg_adherence:.0f}%",
            delta_color="normal"
        )
    
    # Adherence progress bars
    st.subheader("Goal Adherence")
    
    for nutrient, percentage in adherence.items():
        progress_color = "green" if 90 <= percentage <= 110 else "orange" if 80 <= percentage <= 120 else "red"
        
        # Create progress bar
        st.write(f"**{nutrient.title()}:** {percentage:.1f}%")
        progress_html = f"""
        <div style="background-color: #f0f0f0; border-radius: 10px; padding: 3px;">
            <div style="background-color: {progress_color}; width: {min(percentage, 100)}%; height: 20px; border-radius: 8px; text-align: center; color: white; font-weight: bold;">
                {percentage:.0f}%
            </div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
    
    # Daily breakdown chart
    st.subheader("Daily Calorie Intake")
    
    daily_data = report['daily_data']
    dates = [day.get('date', '') for day in daily_data]
    calories = [day.get('total_calories', 0) for day in daily_data]
    
    fig = go.Figure()
    
    # Add calorie line
    fig.add_trace(go.Scatter(
        x=dates,
        y=calories,
        mode='lines+markers',
        name='Daily Calories',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    # Add target line
    fig.add_hline(
        y=targets['calories'],
        line_dash="dash",
        line_color="red",
        annotation_text="Target"
    )
    
    fig.update_layout(
        title="Daily Calorie Intake vs Target",
        xaxis_title="Date",
        yaxis_title="Calories",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Macro distribution
    st.subheader("Weekly Macro Distribution")
    
    # Calculate total calories from each macro
    total_protein_cal = summary['avg_protein'] * 4 * 7
    total_carbs_cal = summary['avg_carbs'] * 4 * 7
    total_fat_cal = summary['avg_fat'] * 9 * 7
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=['Protein', 'Carbohydrates', 'Fat'],
        values=[total_protein_cal, total_carbs_cal, total_fat_cal],
        hole=0.4,
        marker_colors=['#ff6b6b', '#4ecdc4', '#45b7d1']
    )])
    
    fig_pie.update_layout(title="Macro Distribution (by calories)")
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Insights
    if report['insights']:
        st.subheader("🔍 Insights & Recommendations")
        for insight in report['insights']:
            st.markdown(f"• {insight}")

def show_monthly_report(user_id: int):
    """Show monthly nutrition report"""
    st.header("Monthly Nutrition Report")
    
    # Month selector
    col1, col2 = st.columns(2)
    with col1:
        current_year = datetime.now().year
        year = st.selectbox("Year", range(current_year - 2, current_year + 1), index=2)
    with col2:
        current_month = datetime.now().month
        month = st.selectbox("Month", range(1, 13), index=current_month - 1, 
                           format_func=lambda x: datetime(2000, x, 1).strftime('%B'))
    
    # Generate report
    with st.spinner("Generating monthly report..."):
        report = ReportingService.generate_monthly_report(user_id, year, month)
    
    if not report:
        st.warning("No data available for the selected month.")
        return
    
    # Display report
    st.subheader(f"Report for {datetime(year, month, 1).strftime('%B %Y')}")
    
    summary = report['summary']
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Meals",
            summary['total_meals'],
            f"{summary['avg_meals_per_day']:.1f} per day"
        )
    
    with col2:
        st.metric(
            "Logging Consistency",
            f"{summary['logging_consistency']:.1f}%",
            f"{summary['logged_days']}/{summary['total_days']} days"
        )
    
    with col3:
        st.metric(
            "Avg Daily Calories",
            f"{summary['avg_calories']:.0f}",
            help="Average calories consumed per day"
        )
    
    with col4:
        adherence = report['adherence']
        avg_adherence = sum(adherence.values()) / len(adherence)
        st.metric(
            "Avg Goal Adherence",
            f"{avg_adherence:.0f}%"
        )
    
    # Monthly calendar heatmap
    st.subheader("Daily Calorie Heatmap")
    
    daily_totals = report['daily_totals']
    if daily_totals:
        # Create calendar heatmap data
        cal_data = []
        for day, data in daily_totals.items():
            cal_data.append({
                'date': day,
                'calories': data['calories'],
                'day_of_week': day.weekday(),
                'week': day.isocalendar()[1]
            })
        
        if cal_data:
            df_cal = pd.DataFrame(cal_data)
            
            # Create heatmap
            fig_heatmap = px.density_heatmap(
                df_cal,
                x='week',
                y='day_of_week',
                z='calories',
                title="Daily Calorie Intake Calendar",
                color_continuous_scale='RdYlBu_r'
            )
            
            fig_heatmap.update_layout(
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(7)),
                    ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                ),
                xaxis_title="Week of Year",
                yaxis_title="Day of Week"
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Monthly trends
    st.subheader("Monthly Nutrition Trends")
    
    # This would show trends within the month if we had daily aggregated data
    # For now, show summary stats
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Macro breakdown
        macros_data = {
            'Nutrient': ['Protein', 'Carbohydrates', 'Fat'],
            'Daily Average (g)': [
                summary['avg_protein'],
                summary['avg_carbs'],
                summary['avg_fat']
            ],
            'Target (g)': [
                report['targets']['protein'],
                report['targets']['carbs'],
                report['targets']['fat']
            ]
        }
        
        df_macros = pd.DataFrame(macros_data)
        
        fig_macros = px.bar(
            df_macros,
            x='Nutrient',
            y=['Daily Average (g)', 'Target (g)'],
            barmode='group',
            title="Average Daily Macros vs Targets"
        )
        
        st.plotly_chart(fig_macros, use_container_width=True)
    
    with col2:
        # Adherence radar chart
        adherence = report['adherence']
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=list(adherence.values()),
            theta=list(adherence.keys()),
            fill='toself',
            name='Adherence %'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 150]
                )),
            showlegend=False,
            title="Goal Adherence Radar"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Monthly insights
    if report['insights']:
        st.subheader("🔍 Monthly Insights")
        for insight in report['insights']:
            st.markdown(f"• {insight}")

def show_progress_tracking(user_id: int):
    """Show progress tracking over time"""
    st.header("Progress Tracking")
    
    # Time period selector
    weeks = st.selectbox("Time Period", [4, 8, 12, 16], index=1, format_func=lambda x: f"Last {x} weeks")
    
    # Generate progress report
    with st.spinner("Analyzing progress..."):
        progress_report = ReportingService.generate_progress_report(user_id, weeks)
    
    if not progress_report:
        st.warning("Insufficient data for progress tracking.")
        return
    
    # Progress overview
    st.subheader(f"Progress Over {progress_report['period']}")
    
    weekly_reports = progress_report['weekly_reports']
    trends = progress_report['trends']
    
    if not weekly_reports:
        st.warning("No weekly data available.")
        return
    
    # Create progress charts
    weeks_data = []
    for week_report in weekly_reports:
        week_data = week_report['data']['weekly_averages']
        weeks_data.append({
            'week': f"Week {week_report['week']}",
            'start_date': week_report['start_date'],
            'calories': week_data['avg_calories'],
            'protein': week_data['avg_protein'],
            'carbs': week_data['avg_carbs'],
            'fat': week_data['avg_fat'],
            'meals': week_data['avg_meals_per_day']
        })
    
    df_progress = pd.DataFrame(weeks_data)
    
    # Multi-line chart for macros
    st.subheader("Nutrition Trends Over Time")
    
    fig_trends = go.Figure()
    
    fig_trends.add_trace(go.Scatter(
        x=df_progress['week'],
        y=df_progress['calories'],
        mode='lines+markers',
        name='Calories',
        yaxis='y'
    ))
    
    fig_trends.add_trace(go.Scatter(
        x=df_progress['week'],
        y=df_progress['protein'],
        mode='lines+markers',
        name='Protein (g)',
        yaxis='y2'
    ))
    
    fig_trends.add_trace(go.Scatter(
        x=df_progress['week'],
        y=df_progress['carbs'],
        mode='lines+markers',
        name='Carbs (g)',
        yaxis='y2'
    ))
    
    fig_trends.add_trace(go.Scatter(
        x=df_progress['week'],
        y=df_progress['fat'],
        mode='lines+markers',
        name='Fat (g)',
        yaxis='y2'
    ))
    
    # Create subplots with secondary y-axis
    fig_trends.update_layout(
        title="Weekly Nutrition Trends",
        xaxis_title="Week",
        yaxis=dict(title="Calories", side="left"),
        yaxis2=dict(title="Grams", side="right", overlaying="y"),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Trend analysis
    if trends:
        st.subheader("📈 Trend Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Weekly Changes:**")
            for nutrient, change in trends.items():
                if abs(change) > 1:  # Only show significant changes
                    direction = "📈" if change > 0 else "📉"
                    st.markdown(f"{direction} {nutrient.title()}: {change:+.1f}% per week")
        
        with col2:
            # Show meals per day trend
            st.markdown("**Logging Consistency:**")
            avg_meals = df_progress['meals'].mean()
            latest_meals = df_progress['meals'].iloc[-1] if len(df_progress) > 0 else 0
            st.metric("Avg Meals/Day", f"{avg_meals:.1f}", f"{latest_meals:.1f} this week")
    
    # Progress insights
    if 'insights' in progress_report and progress_report['insights']:
        st.subheader("🔍 Progress Insights")
        for insight in progress_report['insights']:
            st.markdown(f"• {insight}")
    
    # Goal achievement summary
    st.subheader("🎯 Goal Achievement Summary")
    
    # This would show how well the user is meeting their goals over time
    st.info("Goal achievement tracking will be implemented based on user preference targets.")

if __name__ == "__main__":
    show_reports_page()