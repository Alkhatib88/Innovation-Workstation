import bcrypt
import json

class UserSystem:
    def __init__(self, database):
        self.db = database

    def hash_password(self, password):
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    def check_password(self, hashed_password, user_password):
        return bcrypt.checkpw(user_password.encode(), hashed_password)

    def register_user(self, username, password, role='user'):
        hashed_password = self.hash_password(password)
        # Add user to the database with role
        # self.db.add_user(username, hashed_password, role)
        return True

    def authenticate_user(self, username, password):
        # Fetch user from the database
        # user = self.db.get_user(username)
        user = {'username': username, 'password': 'hashed_password', 'role': 'user'}  # Example user
        if user and self.check_password(user['password'].encode(), password):
            return user
        return None

    def check_role(self, user, role):
        # Check if the user has the specified role
        return user.get('role') == role

    def lock_account(self, username):
        # Logic to lock the user account
        pass

    def unlock_account(self, username):
        # Logic to unlock the user account
        pass

    # Additional methods for user profile management, email verification, etc.
