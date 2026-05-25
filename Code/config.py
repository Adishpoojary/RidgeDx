"""
RidgeDx — Environment-based Configuration

Usage:
    from config import get_config
    app.config.from_object(get_config())

Environment variables:
    FLASK_ENV: 'development' | 'production' | 'testing'
    SECRET_KEY: Flask secret key (auto-generated if not set)
    DATABASE_URL: Path to SQLite database
"""

import os
import secrets


class Config:
    """Base configuration."""
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(32))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload
    DATABASE = os.environ.get('DATABASE_URL', os.path.join(BASE_DIR, 'users.db'))
    MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(BASE_DIR, 'resnet18_bloodgroup.pth'))
    BLOOD_GROUPS = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
    NUM_CLASSES = 8
    IMAGE_SIZE = 128
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*')


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    # In production, SECRET_KEY must be set via environment
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'https://ridgedx.com')


class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    DATABASE = ':memory:'


_config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}


def get_config():
    """Return config class based on FLASK_ENV environment variable."""
    env = os.environ.get('FLASK_ENV', 'development').lower()
    return _config_map.get(env, DevelopmentConfig)
