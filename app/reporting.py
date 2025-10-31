"""
Reporting service for generating nutrition and meal tracking reports.
"""
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from app.meal_tracking import MealTrackingService
from app.auth import AuthService

logger = logging.getLogger(__name__)

class ReportingService:
    """Handles report generation and analytics"""
    
    @staticmethod
    def generate_weekly_report(user_id: int, start_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate comprehensive weekly nutrition report"""
        try:
            # Get user preferences for targets
            user_prefs = AuthService.get_user_preferences(user_id)
            if not user_prefs:
                logger.error(f"User preferences not found for user {user_id}")
                return {}
            
            # Get weekly data
            weekly_data = MealTrackingService.get_weekly_nutrition_summary(user_id, start_date)
            if not weekly_data:
                return {}
            
            # Calculate target vs actual
            calorie_target = user_prefs.get('calorie_target', 2500)
            protein_target = (calorie_target * user_prefs.get('protein_target_pct', 25) / 100) / 4  # 4 cal per g
            carb_target = (calorie_target * user_prefs.get('carb_target_pct', 45) / 100) / 4
            fat_target = (calorie_target * user_prefs.get('fat_target_pct', 30) / 100) / 9  # 9 cal per g
            
            averages = weekly_data['weekly_averages']
            
            # Calculate adherence percentages
            adherence = {
                'calories': (averages['avg_calories'] / calorie_target) * 100 if calorie_target > 0 else 0,
                'protein': (averages['avg_protein'] / protein_target) * 100 if protein_target > 0 else 0,
                'carbs': (averages['avg_carbs'] / carb_target) * 100 if carb_target > 0 else 0,
                'fat': (averages['avg_fat'] / fat_target) * 100 if fat_target > 0 else 0,
            }
            
            # Generate insights
            insights = ReportingService._generate_weekly_insights(weekly_data, user_prefs, adherence)
            
            # Create charts data
            charts = ReportingService._create_weekly_charts(weekly_data, {
                'calories': calorie_target,
                'protein': protein_target,
                'carbs': carb_target,
                'fat': fat_target
            })
            
            return {
                'period': f"{weekly_data['start_date'].date()} to {weekly_data['end_date'].date()}",
                'summary': {
                    'total_meals': weekly_data['weekly_totals']['meals'],
                    'avg_meals_per_day': round(averages['avg_meals_per_day'], 1),
                    'avg_calories': round(averages['avg_calories'], 0),
                    'avg_protein': round(averages['avg_protein'], 1),
                    'avg_carbs': round(averages['avg_carbs'], 1),
                    'avg_fat': round(averages['avg_fat'], 1),
                },
                'targets': {
                    'calories': calorie_target,
                    'protein': round(protein_target, 1),
                    'carbs': round(carb_target, 1),
                    'fat': round(fat_target, 1),
                },
                'adherence': {k: round(v, 1) for k, v in adherence.items()},
                'insights': insights,
                'charts': charts,
                'daily_data': weekly_data['daily_data']
            }
            
        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")
            return {}
    
    @staticmethod
    def generate_monthly_report(user_id: int, year: Optional[int] = None, month: Optional[int] = None) -> Dict[str, Any]:
        """Generate comprehensive monthly nutrition report"""
        try:
            # Get user preferences
            user_prefs = AuthService.get_user_preferences(user_id)
            if not user_prefs:
                return {}
            
            # Get monthly data
            monthly_data = MealTrackingService.get_monthly_nutrition_summary(user_id, year, month)
            if not monthly_data:
                return {}
            
            # Calculate targets (same as weekly)
            calorie_target = user_prefs.get('calorie_target', 2500)
            protein_target = (calorie_target * user_prefs.get('protein_target_pct', 25) / 100) / 4
            carb_target = (calorie_target * user_prefs.get('carb_target_pct', 45) / 100) / 4
            fat_target = (calorie_target * user_prefs.get('fat_target_pct', 30) / 100) / 9
            
            averages = monthly_data['monthly_averages']
            
            # Calculate adherence
            adherence = {
                'calories': (averages['avg_calories'] / calorie_target) * 100 if calorie_target > 0 else 0,
                'protein': (averages['avg_protein'] / protein_target) * 100 if protein_target > 0 else 0,
                'carbs': (averages['avg_carbs'] / carb_target) * 100 if carb_target > 0 else 0,
                'fat': (averages['avg_fat'] / fat_target) * 100 if fat_target > 0 else 0,
            }
            
            # Generate insights
            insights = ReportingService._generate_monthly_insights(monthly_data, user_prefs, adherence)
            
            # Create charts
            charts = ReportingService._create_monthly_charts(monthly_data, {
                'calories': calorie_target,
                'protein': protein_target,
                'carbs': carb_target,
                'fat': fat_target
            })
            
            return {
                'period': f"{monthly_data['year']}-{monthly_data['month']:02d}",
                'summary': {
                    'total_meals': monthly_data['monthly_totals']['meals'],
                    'logged_days': monthly_data['logged_days'],
                    'total_days': monthly_data['total_days'],
                    'logging_consistency': round((monthly_data['logged_days'] / monthly_data['total_days']) * 100, 1),
                    'avg_meals_per_day': round(averages['avg_meals_per_day'], 1),
                    'avg_calories': round(averages['avg_calories'], 0),
                    'avg_protein': round(averages['avg_protein'], 1),
                    'avg_carbs': round(averages['avg_carbs'], 1),
                    'avg_fat': round(averages['avg_fat'], 1),
                },
                'targets': {
                    'calories': calorie_target,
                    'protein': round(protein_target, 1),
                    'carbs': round(carb_target, 1),
                    'fat': round(fat_target, 1),
                },
                'adherence': {k: round(v, 1) for k, v in adherence.items()},
                'insights': insights,
                'charts': charts,
                'daily_totals': monthly_data['daily_totals']
            }
            
        except Exception as e:
            logger.error(f"Error generating monthly report: {e}")
            return {}
    
    @staticmethod
    def generate_progress_report(user_id: int, weeks: int = 4) -> Dict[str, Any]:
        """Generate progress report over multiple weeks"""
        try:
            # Get data for multiple weeks
            weekly_reports = []
            today = datetime.now(timezone.utc).date()
            
            for i in range(weeks):
                week_start = today - timedelta(days=today.weekday() + (i * 7))
                week_start_dt = datetime.combine(week_start, datetime.min.time()).replace(tzinfo=timezone.utc)
                weekly_data = MealTrackingService.get_weekly_nutrition_summary(user_id, week_start_dt)
                if weekly_data:
                    weekly_reports.append({
                        'week': i + 1,
                        'start_date': week_start,
                        'data': weekly_data
                    })
            
            if not weekly_reports:
                return {}
            
            # Calculate trends
            trends = ReportingService._calculate_trends(weekly_reports)
            
            # Generate progress insights
            insights = ReportingService._generate_progress_insights(weekly_reports, trends)
            
            return {
                'period': f"Last {weeks} weeks",
                'weekly_reports': weekly_reports,
                'trends': trends,
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error generating progress report: {e}")
            return {}
    
    @staticmethod
    def _generate_weekly_insights(weekly_data: Dict, user_prefs: Dict, adherence: Dict) -> List[str]:
        """Generate insights for weekly report"""
        insights = []
        
        # Calorie adherence
        cal_adherence = adherence['calories']
        if cal_adherence < 80:
            insights.append(f"⚠️ You're consuming {100-cal_adherence:.0f}% fewer calories than your target. Consider adding more meals.")
        elif cal_adherence > 120:
            insights.append(f"⚠️ You're consuming {cal_adherence-100:.0f}% more calories than your target. Consider smaller portions.")
        else:
            insights.append("✅ Great job staying within your calorie targets!")
        
        # Protein intake
        if adherence['protein'] < 80:
            insights.append("🥩 Try to include more protein sources like lean meats, eggs, or legumes.")
        elif adherence['protein'] > 100:
            insights.append("💪 Excellent protein intake! This supports your fitness goals.")
        
        # Meal consistency
        avg_meals = weekly_data['weekly_averages']['avg_meals_per_day']
        if avg_meals < 2:
            insights.append("🍽️ Consider logging more meals to get better nutrition tracking.")
        elif avg_meals >= 3:
            insights.append("📊 Great meal logging consistency!")
        
        # Goal-specific insights
        goal = user_prefs.get('goal', 'Balanced')
        if goal == 'Cutting' and cal_adherence < 95:
            insights.append("🎯 Perfect calorie control for your cutting goals!")
        elif goal == 'Bulking' and adherence['protein'] > 100 and cal_adherence > 105:
            insights.append("💪 Excellent nutrition for muscle building!")
        
        return insights
    
    @staticmethod
    def _generate_monthly_insights(monthly_data: Dict, user_prefs: Dict, adherence: Dict) -> List[str]:
        """Generate insights for monthly report"""
        insights = []
        
        # Logging consistency
        consistency = (monthly_data['logged_days'] / monthly_data['total_days']) * 100
        if consistency < 50:
            insights.append("📱 Try to log meals more consistently for better tracking.")
        elif consistency > 80:
            insights.append("🏆 Excellent logging consistency this month!")
        
        # Monthly trends would go here
        # This could compare to previous months, seasonal patterns, etc.
        
        return insights
    
    @staticmethod
    def _generate_progress_insights(weekly_reports: List, trends: Dict) -> List[str]:
        """Generate insights for progress report"""
        insights = []
        
        # Trend analysis
        for nutrient, trend in trends.items():
            if abs(trend) > 5:  # Significant change
                direction = "increasing" if trend > 0 else "decreasing"
                insights.append(f"📈 Your {nutrient} intake is {direction} by {abs(trend):.1f}% per week.")
        
        return insights
    
    @staticmethod
    def _calculate_trends(weekly_reports: List) -> Dict[str, float]:
        """Calculate trends across weeks"""
        trends = {}
        
        try:
            # Extract data for trend calculation
            weeks = len(weekly_reports)
            if weeks < 2:
                return trends
            
            # Calculate linear trends (simple slope)
            for nutrient in ['calories', 'protein', 'carbs', 'fat']:
                values = []
                for report in reversed(weekly_reports):  # Most recent first
                    avg_key = f'avg_{nutrient}'
                    value = report['data']['weekly_averages'].get(avg_key, 0)
                    values.append(value)
                
                if len(values) >= 2:
                    # Simple linear trend (percentage change per week)
                    first_val = values[-1]  # Oldest
                    last_val = values[0]   # Most recent
                    if first_val > 0:
                        percent_change = ((last_val - first_val) / first_val) * 100
                        weekly_change = percent_change / (weeks - 1)
                        trends[nutrient] = weekly_change
            
        except Exception as e:
            logger.error(f"Error calculating trends: {e}")
        
        return trends
    
    @staticmethod
    def _create_weekly_charts(weekly_data: Dict, targets: Dict) -> Dict[str, Any]:
        """Create chart data for weekly report"""
        charts = {}
        
        try:
            # Daily calories chart
            daily_data = weekly_data['daily_data']
            dates = [day.get('date', '') for day in daily_data]
            calories = [day.get('total_calories', 0) for day in daily_data]
            
            charts['daily_calories'] = {
                'type': 'line',
                'data': {
                    'dates': dates,
                    'calories': calories,
                    'target': targets['calories']
                }
            }
            
            # Macro distribution pie chart
            totals = weekly_data['weekly_totals']
            charts['macro_distribution'] = {
                'type': 'pie',
                'data': {
                    'labels': ['Protein', 'Carbs', 'Fat'],
                    'values': [totals['protein'] * 4, totals['carbs'] * 4, totals['fat'] * 9]
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating weekly charts: {e}")
        
        return charts
    
    @staticmethod
    def _create_monthly_charts(monthly_data: Dict, targets: Dict) -> Dict[str, Any]:
        """Create chart data for monthly report"""
        charts = {}
        
        try:
            # Daily calories heatmap
            daily_totals = monthly_data['daily_totals']
            charts['monthly_heatmap'] = {
                'type': 'heatmap',
                'data': daily_totals
            }
            
        except Exception as e:
            logger.error(f"Error creating monthly charts: {e}")
        
        return charts