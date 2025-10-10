"""
Storage utilities for managing persistent data like meals and user profiles.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)

class StorageError(Exception):
    """Base class for storage-related errors."""
    pass

def get_data_path(filename: str) -> Path:
    """Get the absolute path to a data file."""
    return Path(__file__).parent.parent / "data" / filename

def load_json(filename: str) -> Dict:
    """
    Load data from a JSON file in the data directory.
    
    Args:
        filename: Name of the JSON file (e.g., 'meals.json')
        
    Returns:
        Dict containing the loaded data
        
    Raises:
        StorageError: If file doesn't exist or contains invalid JSON
    """
    try:
        filepath = get_data_path(filename)
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Data file not found: {filename}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filename}: {e}")
        raise StorageError(f"Invalid JSON in {filename}") from e

def save_json(filename: str, data: Dict) -> None:
    """
    Save data to a JSON file in the data directory.
    
    Args:
        filename: Name of the JSON file (e.g., 'meals.json')
        data: Dictionary to save
        
    Raises:
        StorageError: If data can't be saved
    """
    try:
        filepath = get_data_path(filename)
        # Ensure data directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    except (OSError, TypeError) as e:
        logger.error(f"Error saving {filename}: {e}")
        raise StorageError(f"Could not save {filename}") from e

def load_user_profile(user_id: str = "default") -> Optional[Dict]:
    """
    Load a specific user profile from profiles.json.
    
    Args:
        user_id: ID of the user profile to load
        
    Returns:
        Dict containing user profile data, or None if not found
    """
    try:
        profiles = load_json("user_profiles.json")
        return profiles.get(user_id)
    except StorageError:
        logger.error(f"Could not load profile for user: {user_id}")
        return None

def save_user_profile(user_id: str, profile_data: Dict) -> bool:
    """
    Save or update a user profile in profiles.json.
    
    Args:
        user_id: ID of the user profile to save
        profile_data: Dictionary containing profile data
        
    Returns:
        bool indicating success
    """
    try:
        profiles = load_json("user_profiles.json")
        profiles[user_id] = profile_data
        save_json("user_profiles.json", profiles)
        return True
    except StorageError:
        logger.error(f"Could not save profile for user: {user_id}")
        return False

def get_meal_nutrition(dish_name: str) -> Optional[Dict]:
    """
    Get cached nutrition data for a dish if available.
    
    Args:
        dish_name: Name of the dish to look up
        
    Returns:
        Dict with nutrition data if found, None otherwise
    """
    try:
        meals = load_json("meals.json")
        return meals.get(dish_name.lower())
    except StorageError:
        logger.error(f"Could not load nutrition data for: {dish_name}")
        return None

def save_meal_nutrition(dish_name: str, nutrition_data: Dict) -> bool:
    """
    Save or update nutrition data for a dish.
    
    Args:
        dish_name: Name of the dish
        nutrition_data: Dictionary with nutrition information
        
    Returns:
        bool indicating success
    """
    try:
        meals = load_json("meals.json")
        meals[dish_name.lower()] = nutrition_data
        save_json("meals.json", meals)
        return True
    except StorageError:
        logger.error(f"Could not save nutrition data for: {dish_name}")
        return False

def append_meal_history(
    user_id: str,
    dish_name: str,
    nutrition: Dict,
    timestamp: Optional[str] = None
) -> bool:
    """
    Add a meal to the user's meal history.
    
    Args:
        user_id: ID of the user
        dish_name: Name of the dish eaten
        nutrition: Dictionary of nutrition information
        timestamp: Optional timestamp (ISO format)
        
    Returns:
        bool indicating success
    """
    if timestamp is None:
        from datetime import datetime
        timestamp = datetime.utcnow().isoformat()
        
    try:
        history_file = f"meal_history_{user_id}.json"
        try:
            history = load_json(history_file)
        except StorageError:
            history = []
            
        if not isinstance(history, list):
            history = []
            
        meal_entry = {
            "timestamp": timestamp,
            "dish_name": dish_name,
            "nutrition": nutrition
        }
        
        history.append(meal_entry)
        save_json(history_file, history)
        return True
        
    except StorageError:
        logger.error(f"Could not save meal history for user: {user_id}")
        return False

def get_meal_history(
    user_id: str,
    limit: Optional[int] = None
) -> list:
    """
    Get the meal history for a user.
    
    Args:
        user_id: ID of the user
        limit: Optional maximum number of entries to return
        
    Returns:
        List of meal entries, sorted by timestamp (newest first)
    """
    try:
        history = load_json(f"meal_history_{user_id}.json")
        if not isinstance(history, list):
            return []
            
        # Sort by timestamp, newest first
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        if limit:
            return history[:limit]
        return history
        
    except StorageError:
        logger.error(f"Could not load meal history for user: {user_id}")
        return []
