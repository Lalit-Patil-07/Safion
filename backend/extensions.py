"""
Shared extension singletons.
Import these into modules that need them — never import `app` directly.
The app factory (app.py) calls init_app() on each during startup.
"""
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager

db = SQLAlchemy()
bcrypt = Bcrypt()
jwt = JWTManager()
