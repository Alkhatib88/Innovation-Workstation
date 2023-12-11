import bcrypt
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer

class AuthenticationService:
    def __init__(self, user_system, secret_key, token_expiration=3600):
        self.user_system = user_system
        self.secret_key = secret_key
        self.token_expiration = token_expiration

    def authenticate_user(self, username, password):
        user = self.user_system.get_user(username)
        if user and bcrypt.checkpw(password.encode(), user['password'].encode()):
            return user
        return None

    def generate_token(self, user):
        s = Serializer(self.secret_key, expires_in=self.token_expiration)
        return s.dumps({'user_id': user['id']}).decode('utf-8')

    def verify_token(self, token):
        s = Serializer(self.secret_key)
        try:
            data = s.loads(token)
            user_id = data['user_id']
            return self.user_system.get_user_by_id(user_id)
        except:
            return None

# Example usage
# user_system = UserSystem(db)  # Assuming a UserSystem instance
# auth_service = AuthenticationService(user_system, 'YOUR_SECRET_KEY')
# user = auth_service.authenticate_user('username', 'password')
# if user:
#     token = auth_service.generate_token(user)
#     print("Auth Token:", token)
