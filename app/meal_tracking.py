"""
Meal tracking service for logging and managing user meals.
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc
from app.database import UserMeal, NutritionGoal, get_db, close_db

logger = logging.getLogger(__name__)

class MealTrackingService:
    """Handles meal logging, confirmation, and tracking"""
    
    @staticmethod
    def save_meal_analysis(
        user_id: int,
        meal_name: str,
        nutrition_data: Dict[str, Any],
        components: List[Dict[str, Any]],
        image_hash: str,
        confidence_score: float,
        ai_suggestions: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        health_score: Optional[int] = None,
        consumed_at: Optional[datetime] = None
    ) -> Optional[UserMeal]:
        """Save a meal analysis (not yet confirmed as eaten)"""
        db = get_db()
        try:
            if consumed_at is None:
                consumed_at = datetime.now(timezone.utc)
            
            # Determine meal type based on time
            meal_type = MealTrackingService._determine_meal_type(consumed_at)
            
            meal = UserMeal(
                user_id=user_id,
                meal_name=meal_name,
                meal_type=meal_type,
                nutrition_data=nutrition_data,
                components=components,
                image_hash=image_hash,
                confidence_score=confidence_score,
                consumed_at=consumed_at,
                health_score=health_score,
                ai_suggestions=ai_suggestions or [],
                warnings=warnings or [],
                confirmed=False  # Not confirmed yet
            )
            
            db.add(meal)
            db.commit()
            db.refresh(meal)
            
            logger.info(f"Meal analysis saved for user {user_id}: {meal_name}")
            return meal
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving meal analysis: {e}")
            return None
        finally:
            close_db(db)
    
    @staticmethod
    def confirm_meal(meal_id: int, user_id: int, rating: Optional[int] = None, notes: Optional[str] = None) -> bool:
        """Confirm that a meal was actually consumed"""
        db = get_db()
        try:
            meal = db.query(UserMeal).filter(
                and_(UserMeal.id == meal_id, UserMeal.user_id == user_id)
            ).first()
            
            if not meal:
                logger.warning(f"Meal not found for confirmation: {meal_id}")
                return False
            
            meal.confirmed = True  # type: ignore
            meal.rating = rating  # type: ignore
            meal.notes = notes  # type: ignore
            
            db.commit()
            
            # Update daily nutrition goals
            MealTrackingService._update_daily_nutrition(user_id, meal.consumed_at, db)  # type: ignore
            
            logger.info(f"Meal confirmed for user {user_id}: {meal.meal_name}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error confirming meal: {e}")
            return False
        finally:
            close_db(db)
    
    @staticmethod
    def get_recent_meals(user_id: int, days: int = 7, confirmed_only: bool = True) -> List[UserMeal]:
        """Get recent meals for a user"""
        db = get_db()
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            query = db.query(UserMeal).filter(
                and_(
                    UserMeal.user_id == user_id,
                    UserMeal.consumed_at >= cutoff_date
                )
            )
            
            if confirmed_only:
                query = query.filter(UserMeal.confirmed == True)
            
            meals = query.order_by(desc(UserMeal.consumed_at)).all()
            return meals
            
        except Exception as e:
            logger.error(f"Error getting recent meals: {e}")
            return []
        finally:
            close_db(db)
    
    @staticmethod
    def get_daily_nutrition_summary(user_id: int, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get nutrition summary for a specific day"""
        if date is None:
            date = datetime.now(timezone.utc)
        
        db = get_db()
        try:
            # Get date range for the day
            start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = start_date + timedelta(days=1)
            
            # Get confirmed meals for the day
            meals = db.query(UserMeal).filter(
                and_(
                    UserMeal.user_id == user_id,
                    UserMeal.confirmed == True,
                    UserMeal.consumed_at >= start_date,
                    UserMeal.consumed_at < end_date
                )
            ).all()
            
            # Calculate totals
            total_calories = 0
            total_protein = 0.0
            total_carbs = 0.0
            total_fat = 0.0
            total_fiber = 0.0
            
            meal_breakdown = []
            
            for meal in meals:
                nutrition = meal.nutrition_data.get('nutrition_summary', {})
                calories = nutrition.get('calories', {}).get('value', 0)
                macros = nutrition.get('macros', {})
                additional = nutrition.get('additional', {})
                
                protein = macros.get('protein', {}).get('value', 0)
                carbs = macros.get('carbs', {}).get('value', 0)
                fat = macros.get('fat', {}).get('value', 0)
                fiber = additional.get('fiber', {}).get('value', 0)
                
                total_calories += calories
                total_protein += protein
                total_carbs += carbs
                total_fat += fat
                total_fiber += fiber
                
                meal_breakdown.append({
                    'id': meal.id,
                    'name': meal.meal_name,
                    'type': meal.meal_type,
                    'time': meal.consumed_at,
                    'calories': calories,
                    'protein': protein,
                    'carbs': carbs,
                    'fat': fat,
                    'fiber': fiber,
                    'rating': meal.rating,
                    'health_score': meal.health_score
                })
            
            return {
                'date': date.date(),
                'total_calories': total_calories,
                'total_protein': total_protein,
                'total_carbs': total_carbs,
                'total_fat': total_fat,
                'total_fiber': total_fiber,
                'meal_count': len(meals),
                'meals': meal_breakdown
            }
            
        except Exception as e:
            logger.error(f"Error getting daily nutrition summary: {e}")
            return {}
        finally:
            close_db(db)
    
    @staticmethod
    def get_weekly_nutrition_summary(user_id: int, start_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get nutrition summary for a week"""
        if start_date is None:
            today = datetime.now(timezone.utc).date()
            start_date = datetime.combine(
                today - timedelta(days=today.weekday()), 
                datetime.min.time()
            ).replace(tzinfo=timezone.utc)  # Start of week (Monday)
        
        weekly_data = []
        weekly_totals = {
            'calories': 0,
            'protein': 0.0,
            'carbs': 0.0,
            'fat': 0.0,
            'fiber': 0.0,
            'meals': 0
        }
        
        for i in range(7):
            day = start_date + timedelta(days=i)
            day_summary = MealTrackingService.get_daily_nutrition_summary(user_id, day)
            
            weekly_data.append(day_summary)
            
            # Add to weekly totals
            weekly_totals['calories'] += day_summary.get('total_calories', 0)
            weekly_totals['protein'] += day_summary.get('total_protein', 0)
            weekly_totals['carbs'] += day_summary.get('total_carbs', 0)
            weekly_totals['fat'] += day_summary.get('total_fat', 0)
            weekly_totals['fiber'] += day_summary.get('total_fiber', 0)
            weekly_totals['meals'] += day_summary.get('meal_count', 0)
        
        # Calculate averages using only days with non-zero data
        valid_days = [d for d in weekly_data if d.get('meal_count', 0) > 0]
        num_valid_days = len(valid_days) if valid_days else 1
        weekly_averages = {
            'avg_calories': weekly_totals['calories'] / num_valid_days,
            'avg_protein': weekly_totals['protein'] / num_valid_days,
            'avg_carbs': weekly_totals['carbs'] / num_valid_days,
            'avg_fat': weekly_totals['fat'] / num_valid_days,
            'avg_fiber': weekly_totals['fiber'] / num_valid_days,
            'avg_meals_per_day': weekly_totals['meals'] / num_valid_days
        }
        
        return {
            'start_date': start_date,
            'end_date': start_date + timedelta(days=6),
            'daily_data': weekly_data,
            'weekly_totals': weekly_totals,
            'weekly_averages': weekly_averages
        }
    
    @staticmethod
    def get_monthly_nutrition_summary(user_id: int, year: Optional[int] = None, month: Optional[int] = None) -> Dict[str, Any]:
        """Get nutrition summary for a month"""
        if year is None or month is None:
            now = datetime.now(timezone.utc)
            year = now.year
            month = now.month
        
        # Get first and last day of month
        first_day = datetime(year, month, 1, tzinfo=timezone.utc)
        if month == 12:
            last_day = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
        else:
            last_day = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
        
        db = get_db()
        try:
            # Get all confirmed meals for the month
            meals = db.query(UserMeal).filter(
                and_(
                    UserMeal.user_id == user_id,
                    UserMeal.confirmed == True,
                    UserMeal.consumed_at >= first_day,
                    UserMeal.consumed_at <= last_day
                )
            ).order_by(UserMeal.consumed_at).all()
            
            # Group by day and calculate totals
            daily_totals = {}
            monthly_totals = {
                'calories': 0,
                'protein': 0.0,
                'carbs': 0.0,
                'fat': 0.0,
                'fiber': 0.0,
                'meals': 0
            }
            
            for meal in meals:
                day = meal.consumed_at.date()
                if day not in daily_totals:
                    daily_totals[day] = {
                        'calories': 0,
                        'protein': 0.0,
                        'carbs': 0.0,
                        'fat': 0.0,
                        'fiber': 0.0,
                        'meals': 0
                    }
                
                nutrition = meal.nutrition_data.get('nutrition_summary', {})
                calories = nutrition.get('calories', {}).get('value', 0)
                macros = nutrition.get('macros', {})
                additional = nutrition.get('additional', {})
                
                protein = macros.get('protein', {}).get('value', 0)
                carbs = macros.get('carbs', {}).get('value', 0)
                fat = macros.get('fat', {}).get('value', 0)
                fiber = additional.get('fiber', {}).get('value', 0)
                
                daily_totals[day]['calories'] += calories
                daily_totals[day]['protein'] += protein
                daily_totals[day]['carbs'] += carbs
                daily_totals[day]['fat'] += fat
                daily_totals[day]['fiber'] += fiber
                daily_totals[day]['meals'] += 1
                
                monthly_totals['calories'] += calories
                monthly_totals['protein'] += protein
                monthly_totals['carbs'] += carbs
                monthly_totals['fat'] += fat
                monthly_totals['fiber'] += fiber
                monthly_totals['meals'] += 1
            
            # Calculate averages using only days with logged meals
            logged_days = len(daily_totals)
            divisor = logged_days if logged_days > 0 else 1
            monthly_averages = {
                'avg_calories': monthly_totals['calories'] / divisor,
                'avg_protein': monthly_totals['protein'] / divisor,
                'avg_carbs': monthly_totals['carbs'] / divisor,
                'avg_fat': monthly_totals['fat'] / divisor,
                'avg_fiber': monthly_totals['fiber'] / divisor,
                'avg_meals_per_day': monthly_totals['meals'] / divisor
            }
            
            return {
                'year': year,
                'month': month,
                'first_day': first_day.date(),
                'last_day': last_day.date(),
                'daily_totals': daily_totals,
                'monthly_totals': monthly_totals,
                'monthly_averages': monthly_averages,
                'total_days': divisor,
                'logged_days': len(daily_totals)
            }
            
        except Exception as e:
            logger.error(f"Error getting monthly nutrition summary: {e}")
            return {}
        finally:
            close_db(db)
    
    @staticmethod
    def _determine_meal_type(consumed_at: datetime) -> str:
        """Determine meal type based on time of day"""
        hour = consumed_at.hour
        
        if 5 <= hour < 11:
            return "breakfast"
        elif 11 <= hour < 15:
            return "lunch"
        elif 15 <= hour < 18:
            return "snack"
        elif 18 <= hour < 23:
            return "dinner"
        else:
            return "snack"  # Late night/early morning
    
    @staticmethod
    def _update_daily_nutrition(user_id: int, date: datetime, db: Session):
        """Update daily nutrition goals tracking"""
        try:
            # This would update the NutritionGoal table
            # Implementation depends on how you want to track daily goals
            pass
        except Exception as e:
            logger.error(f"Error updating daily nutrition: {e}")
    
    @staticmethod
    def delete_meal(meal_id: int, user_id: int) -> bool:
        """Delete a meal (only if it belongs to the user)"""
        db = get_db()
        try:
            meal = db.query(UserMeal).filter(
                and_(UserMeal.id == meal_id, UserMeal.user_id == user_id)
            ).first()
            
            if not meal:
                logger.warning(f"Meal not found for deletion: {meal_id}")
                return False
            
            db.delete(meal)
            db.commit()
            
            logger.info(f"Meal deleted for user {user_id}: {meal.meal_name}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting meal: {e}")
            return False
        finally:
            close_db(db)
    
    @staticmethod
    def get_meal_by_image_hash(user_id: int, image_hash: str) -> Optional[UserMeal]:
        """Get meal by image hash (to avoid duplicate analysis)"""
        db = get_db()
        try:
            meal = db.query(UserMeal).filter(
                and_(
                    UserMeal.user_id == user_id,
                    UserMeal.image_hash == image_hash
                )
            ).first()
            return meal
        except Exception as e:
            logger.error(f"Error getting meal by image hash: {e}")
            return None
        finally:
            close_db(db)