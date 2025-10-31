"""
Database models and connection for FitPlate AI food tracking system.
"""
import os
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, 
    Text, Boolean, ForeignKey, JSON, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func
import json

logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.expanduser('~')}/fitplate.db")
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    """User account information"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships
    preferences = relationship("UserPreferences", back_populates="user", uselist=False)
    meals = relationship("UserMeal", back_populates="user")
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"

class UserPreferences(Base):
    """User dietary preferences and goals"""
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    
    # Fitness goals
    goal = Column(String(20), default="Balanced")  # Balanced, Bulking, Cutting, Maintenance
    calorie_target = Column(Integer, default=2500)
    
    # Diet preferences
    diet_type = Column(String(20), default="None")  # None, Keto, Vegan, Vegetarian, Mediterranean
    
    # Health conditions (stored as JSON array)
    health_conditions = Column(JSON, default=list)
    
    # Macro targets (percentages)
    protein_target_pct = Column(Float, default=25.0)  # 25% of calories from protein
    carb_target_pct = Column(Float, default=45.0)     # 45% of calories from carbs
    fat_target_pct = Column(Float, default=30.0)      # 30% of calories from fat
    
    # Personal info
    age = Column(Integer)
    gender = Column(String(10))  # Male, Female, Other
    height_cm = Column(Float)
    weight_kg = Column(Float)
    activity_level = Column(String(20), default="Moderate")  # Sedentary, Light, Moderate, Active, Very Active
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="preferences")
    
    def __repr__(self):
        return f"<UserPreferences(user_id={self.user_id}, goal='{self.goal}')>"

class UserMeal(Base):
    """Individual meals logged by users"""
    __tablename__ = "user_meals"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Meal identification
    meal_name = Column(String(200), nullable=False)
    meal_type = Column(String(20))  # breakfast, lunch, dinner, snack
    
    # Analysis data (stored as JSON)
    nutrition_data = Column(JSON, nullable=False)  # Full nutrition analysis
    components = Column(JSON, default=list)        # Individual food items
    
    # AI insights
    health_score = Column(Integer)  # 0-100
    ai_suggestions = Column(JSON, default=list)
    warnings = Column(JSON, default=list)
    
    # Image data
    image_hash = Column(String(64), index=True)  # MD5 hash of the image
    confidence_score = Column(Float)  # AI confidence in analysis
    
    # Timing
    consumed_at = Column(DateTime(timezone=True), nullable=False)  # When user ate it
    logged_at = Column(DateTime(timezone=True), server_default=func.now())  # When logged
    
    # User confirmation
    confirmed = Column(Boolean, default=False)  # User confirmed they ate this
    rating = Column(Integer)  # User rating 1-5 (optional)
    notes = Column(Text)     # User notes about the meal
    
    # Relationships
    user = relationship("User", back_populates="meals")
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_user_consumed_at', 'user_id', 'consumed_at'),
        Index('idx_user_meal_type', 'user_id', 'meal_type'),
        Index('idx_confirmed_meals', 'user_id', 'confirmed'),
    )
    
    def __repr__(self):
        return f"<UserMeal(user_id={self.user_id}, meal_name='{self.meal_name}', consumed_at='{self.consumed_at}')>"

class NutritionGoal(Base):
    """Daily nutrition goals and tracking"""
    __tablename__ = "nutrition_goals"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    date = Column(DateTime(timezone=True), nullable=False)  # Date for these goals
    
    # Daily targets
    calorie_goal = Column(Integer, nullable=False)
    protein_goal_g = Column(Float, nullable=False)
    carb_goal_g = Column(Float, nullable=False)
    fat_goal_g = Column(Float, nullable=False)
    fiber_goal_g = Column(Float, default=25.0)
    
    # Actual consumption (calculated from meals)
    calories_consumed = Column(Integer, default=0)
    protein_consumed_g = Column(Float, default=0.0)
    carb_consumed_g = Column(Float, default=0.0)
    fat_consumed_g = Column(Float, default=0.0)
    fiber_consumed_g = Column(Float, default=0.0)
    
    # Progress tracking
    goal_met = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_user_date', 'user_id', 'date'),
    )
    
    def __repr__(self):
        return f"<NutritionGoal(user_id={self.user_id}, date='{self.date}', calorie_goal={self.calorie_goal})>"

# Database utility functions
def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # Don't close here, let caller handle it

def close_db(db: Session):
    """Close database session"""
    if db:
        db.close()

def init_database():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

# Helper functions for data conversion
def preferences_to_dict(prefs: UserPreferences) -> Dict[str, Any]:
    """Convert UserPreferences to dictionary format for existing code"""
    return {
        'goal': prefs.goal,
        'diet': prefs.diet_type,
        'health_conditions': prefs.health_conditions or [],
        'calorie_target': prefs.calorie_target,
        'protein_target_pct': prefs.protein_target_pct,
        'carb_target_pct': prefs.carb_target_pct,
        'fat_target_pct': prefs.fat_target_pct,
        'age': prefs.age,
        'gender': prefs.gender,
        'height_cm': prefs.height_cm,
        'weight_kg': prefs.weight_kg,
        'activity_level': prefs.activity_level
    }

def dict_to_preferences(user_id: int, prefs_dict: Dict[str, Any]) -> UserPreferences:
    """Convert dictionary to UserPreferences object"""
    return UserPreferences(
        user_id=user_id,
        goal=prefs_dict.get('goal', 'Balanced'),
        calorie_target=prefs_dict.get('calorie_target', 2500),
        diet_type=prefs_dict.get('diet', 'None'),
        health_conditions=prefs_dict.get('health_conditions', []),
        protein_target_pct=prefs_dict.get('protein_target_pct', 25.0),
        carb_target_pct=prefs_dict.get('carb_target_pct', 45.0),
        fat_target_pct=prefs_dict.get('fat_target_pct', 30.0),
        age=prefs_dict.get('age'),
        gender=prefs_dict.get('gender'),
        height_cm=prefs_dict.get('height_cm'),
        weight_kg=prefs_dict.get('weight_kg'),
        activity_level=prefs_dict.get('activity_level', 'Moderate')
    )