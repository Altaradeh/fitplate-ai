"""
Authentication service for FitPlate AI.
"""
import bcrypt
import logging
from typing import Optional, Dict, Any
import streamlit as st
from sqlalchemy.orm import Session
from sqlalchemy import or_

from .database import get_db, close_db, User, UserPreferences, preferences_to_dict, dict_to_preferences

logger = logging.getLogger(__name__)

class AuthService:
    """Handles user authentication and session management"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password for storing in database"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    @staticmethod
    def create_user(username: str, email: str, password: str, 
                   initial_preferences: Optional[Dict[str, Any]] = None) -> Optional[User]:
        """Create a new user account"""
        db = get_db()
        try:
            # Check if user already exists
            existing_user = db.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing_user:
                logger.warning(f"User creation failed: username '{username}' or email '{email}' already exists")
                return None
            
            # Create new user
            hashed_password = AuthService.hash_password(password)
            new_user = User(
                username=username,
                email=email,
                hashed_password=hashed_password
            )
            
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            
            # Create default preferences
            if initial_preferences:
                prefs = dict_to_preferences(new_user.id, initial_preferences)  # type: ignore
            else:
                prefs = UserPreferences(
                    user_id=new_user.id,
                    goal="Balanced",
                    calorie_target=2500,
                    diet_type="None",
                    health_conditions=[]
                )
            
            db.add(prefs)
            db.commit()
            
            # Access attributes while session is active
            user_id = new_user.id
            username = new_user.username  # type: ignore
            
            # Detach from session so it can be used after session closes
            db.expunge(new_user)
            
            logger.info(f"User created successfully: {username}")
            return new_user
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating user: {e}")
            return None
        finally:
            close_db(db)
    
    @staticmethod
    def authenticate_user(username: str, password: str) -> Optional[User]:
        """Authenticate user credentials"""
        db = get_db()
        try:
            user = db.query(User).filter(User.username == username).first()
            
            if user and user.is_active and AuthService.verify_password(password, user.hashed_password):  # type: ignore
                logger.info(f"User authenticated: {username}")
                # Detach from session so it can be used after session closes
                db.expunge(user)
                return user
            
            logger.warning(f"Authentication failed for user: {username}")
            return None
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
        finally:
            close_db(db)
    
    @staticmethod
    def get_user_by_id(user_id: int) -> Optional[User]:
        """Get user by ID"""
        db = get_db()
        try:
            return db.query(User).filter(User.id == user_id).first()
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
        finally:
            close_db(db)
    
    @staticmethod
    def get_user_preferences(user_id: int) -> Optional[Dict[str, Any]]:
        """Get user preferences as dictionary"""
        db = get_db()
        try:
            prefs = db.query(UserPreferences).filter(UserPreferences.user_id == user_id).first()
            return preferences_to_dict(prefs) if prefs else None
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return None
        finally:
            close_db(db)
    
    @staticmethod
    def update_user_preferences(user_id: int, preferences: Dict[str, Any]) -> bool:
        """Update user preferences"""
        db = get_db()
        try:
            prefs = db.query(UserPreferences).filter(UserPreferences.user_id == user_id).first()
            
            if not prefs:
                # Create new preferences if they don't exist
                prefs = dict_to_preferences(user_id, preferences)
                db.add(prefs)
            else:
                # Update existing preferences
                for key, value in preferences.items():
                    if hasattr(prefs, key):
                        setattr(prefs, key, value)
                    elif key == 'diet':
                        prefs.diet_type = value
            
            db.commit()
            logger.info(f"Preferences updated for user {user_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating preferences: {e}")
            return False
        finally:
            close_db(db)
    
    @staticmethod
    def update_user_profile(user_id: int, profile_data: Dict[str, Any]) -> bool:
        """Update user profile information"""
        db = get_db()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            prefs = db.query(UserPreferences).filter(UserPreferences.user_id == user_id).first()
            
            if not user:
                return False
            
            # Update user info
            if 'email' in profile_data:
                user.email = profile_data['email']
            
            # Update preferences with personal info
            if prefs:
                for key in ['age', 'gender', 'height_cm', 'weight_kg', 'activity_level']:
                    if key in profile_data:
                        setattr(prefs, key, profile_data[key])
            
            db.commit()
            logger.info(f"Profile updated for user {user_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating profile: {e}")
            return False
        finally:
            close_db(db)

# Session management for Streamlit
def init_session_state():
    """Initialize authentication session state"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None

def login_user(user: User):
    """Log in a user to the session"""
    st.session_state.authenticated = True
    st.session_state.user_id = user.id
    st.session_state.username = user.username
    logger.info(f"User logged in to session: {user.username}")

def logout_user():
    """Log out the current user"""
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.username = None
    logger.info("User logged out")

def is_authenticated() -> bool:
    """Check if user is authenticated"""
    return st.session_state.get('authenticated', False)

def get_current_user_id() -> Optional[int]:
    """Get current user ID"""
    return st.session_state.get('user_id')

def get_current_username() -> Optional[str]:
    """Get current username"""
    return st.session_state.get('username')