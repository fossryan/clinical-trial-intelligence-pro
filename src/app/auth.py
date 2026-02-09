"""
User Authentication and Tier Management
Handles user login, tier limits, and feature access control
"""

import streamlit as st
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List


class UserTier:
    """User subscription tiers with feature limits"""
    
    FREE = "free"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    ENTERPRISE_PLUS = "enterprise_plus"
    
    TIER_FEATURES = {
        FREE: {
            'name': 'Free',
            'price': 0,
            'predictions_per_month': 5,
            'competitor_tracking': 0,
            'real_time_monitoring': False,
            'protocol_optimizer': False,
            'financial_calculator': True,  # Limited version
            'export_excel': False,
            'export_powerpoint': False,
            'api_access': False,
        },
        PROFESSIONAL: {
            'name': 'Professional',
            'price': 25000,
            'predictions_per_month': -1,  # Unlimited
            'competitor_tracking': 3,
            'real_time_monitoring': True,
            'real_time_monitoring_trials': 10,
            'protocol_optimizer': False,
            'financial_calculator': True,
            'export_excel': True,
            'export_powerpoint': False,
            'api_access': False,
        },
        ENTERPRISE: {
            'name': 'Enterprise',
            'price': 75000,
            'predictions_per_month': -1,
            'competitor_tracking': 10,
            'real_time_monitoring': True,
            'real_time_monitoring_trials': 50,
            'protocol_optimizer': True,
            'financial_calculator': True,
            'export_excel': True,
            'export_powerpoint': True,
            'api_access': True,
            'api_rate_limit': 1000,  # requests per day
        },
        ENTERPRISE_PLUS: {
            'name': 'Enterprise+',
            'price': 150000,
            'predictions_per_month': -1,
            'competitor_tracking': -1,  # Unlimited
            'real_time_monitoring': True,
            'real_time_monitoring_trials': -1,
            'protocol_optimizer': True,
            'financial_calculator': True,
            'export_excel': True,
            'export_powerpoint': True,
            'api_access': True,
            'api_rate_limit': 10000,
            'custom_model_training': True,
            'white_label': True,
        }
    }


class AuthManager:
    """Manages user authentication and session state"""
    
    def __init__(self, users_file: str = "data/users.json"):
        self.users_file = Path(users_file)
        self.users = self._load_users()
        
        # Initialize session state
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'tier' not in st.session_state:
            st.session_state.tier = UserTier.FREE
        if 'monthly_usage' not in st.session_state:
            st.session_state.monthly_usage = {'predictions': 0, 'last_reset': datetime.now().isoformat()}
    
    def _load_users(self) -> Dict:
        """Load users from JSON file"""
        if not self.users_file.exists():
            # Create default demo users
            default_users = {
                'demo': {
                    'password_hash': self._hash_password('demo123'),
                    'tier': UserTier.FREE,
                    'email': 'demo@encinitas.ai',
                    'created_at': datetime.now().isoformat()
                },
                'pro_demo': {
                    'password_hash': self._hash_password('pro123'),
                    'tier': UserTier.PROFESSIONAL,
                    'email': 'pro@encinitas.ai',
                    'created_at': datetime.now().isoformat()
                },
                'enterprise_demo': {
                    'password_hash': self._hash_password('ent123'),
                    'tier': UserTier.ENTERPRISE,
                    'email': 'enterprise@encinitas.ai',
                    'created_at': datetime.now().isoformat()
                }
            }
            
            # Save default users
            self.users_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.users_file, 'w') as f:
                json.dump(default_users, f, indent=2)
            
            return default_users
        
        with open(self.users_file, 'r') as f:
            return json.load(f)
    
    def _save_users(self):
        """Save users to JSON file"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def _hash_password(self, password: str) -> str:
        """Hash password with SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate user credentials"""
        if username not in self.users:
            return False
        
        password_hash = self._hash_password(password)
        if self.users[username]['password_hash'] == password_hash:
            # Set session state
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.tier = self.users[username]['tier']
            
            # Load usage data
            self._load_usage(username)
            
            return True
        
        return False
    
    def logout(self):
        """Logout current user"""
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.tier = UserTier.FREE
        st.session_state.monthly_usage = {'predictions': 0, 'last_reset': datetime.now().isoformat()}
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return st.session_state.get('authenticated', False)
    
    def get_current_tier(self) -> str:
        """Get current user's tier"""
        return st.session_state.get('tier', UserTier.FREE)
    
    def get_tier_features(self, tier: Optional[str] = None) -> Dict:
        """Get features for a tier"""
        tier = tier or self.get_current_tier()
        return UserTier.TIER_FEATURES.get(tier, UserTier.TIER_FEATURES[UserTier.FREE])
    
    def has_feature(self, feature: str) -> bool:
        """Check if current user has access to a feature"""
        features = self.get_tier_features()
        return features.get(feature, False)
    
    def get_feature_limit(self, feature: str) -> int:
        """Get limit for a feature (-1 means unlimited)"""
        features = self.get_tier_features()
        return features.get(feature, 0)
    
    def _load_usage(self, username: str):
        """Load usage data for user"""
        if 'monthly_usage' in self.users[username]:
            usage = self.users[username]['monthly_usage']
            
            # Reset if new month
            last_reset = datetime.fromisoformat(usage['last_reset'])
            if datetime.now().month != last_reset.month:
                usage = {'predictions': 0, 'last_reset': datetime.now().isoformat()}
                self.users[username]['monthly_usage'] = usage
                self._save_users()
            
            st.session_state.monthly_usage = usage
        else:
            st.session_state.monthly_usage = {'predictions': 0, 'last_reset': datetime.now().isoformat()}
    
    def increment_usage(self, feature: str):
        """Increment usage counter for a feature"""
        if not self.is_authenticated():
            return
        
        username = st.session_state.username
        
        if feature == 'predictions':
            st.session_state.monthly_usage['predictions'] += 1
            
            # Save to user file
            if username in self.users:
                self.users[username]['monthly_usage'] = st.session_state.monthly_usage
                self._save_users()
    
    def check_usage_limit(self, feature: str) -> bool:
        """Check if user has exceeded usage limit for a feature"""
        limit = self.get_feature_limit(f'{feature}_per_month')
        
        if limit == -1:  # Unlimited
            return True
        
        current_usage = st.session_state.monthly_usage.get(feature, 0)
        return current_usage < limit
    
    def render_login_page(self):
        """Render login page"""
        st.title("🔐 Login to Encinitas")
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### Welcome Back")
            st.markdown("Login to access your clinical trial intelligence dashboard")
            
            st.markdown("---")
            
            # Demo accounts info
            with st.expander("📋 Demo Accounts (Click to view credentials)"):
                st.markdown("""
                **Free Tier:**
                - Username: `demo`
                - Password: `demo123`
                - Features: 5 predictions/month, limited features
                
                **Professional Tier:**
                - Username: `pro_demo`
                - Password: `pro123`
                - Features: Unlimited predictions, competitor tracking
                
                **Enterprise Tier:**
                - Username: `enterprise_demo`
                - Password: `ent123`
                - Features: All features, API access, exports
                """)
            
            st.markdown("---")
            
            # Login form
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    login_button = st.form_submit_button("🔓 Login", use_container_width=True)
                
                if login_button:
                    if self.authenticate(username, password):
                        st.success(f"✅ Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
            
            st.markdown("---")
            st.markdown("**Don't have an account?**")
            st.info("Contact sales@encinitas.ai for enterprise access")
    
    def render_tier_badge(self):
        """Render tier badge in sidebar"""
        tier = self.get_current_tier()
        tier_info = self.get_tier_features(tier)
        
        if tier == UserTier.FREE:
            st.sidebar.info(f"🆓 **Tier:** {tier_info['name']}")
        elif tier == UserTier.PROFESSIONAL:
            st.sidebar.success(f"💼 **Tier:** {tier_info['name']}")
        elif tier == UserTier.ENTERPRISE:
            st.sidebar.success(f"🏢 **Tier:** {tier_info['name']}")
        elif tier == UserTier.ENTERPRISE_PLUS:
            st.sidebar.success(f"👑 **Tier:** {tier_info['name']}")
        
        # Show usage for free tier
        if tier == UserTier.FREE:
            predictions_used = st.session_state.monthly_usage.get('predictions', 0)
            predictions_limit = tier_info['predictions_per_month']
            st.sidebar.caption(f"Predictions: {predictions_used}/{predictions_limit} this month")
    
    def require_tier(self, required_tier: str, feature_name: str) -> bool:
        """Check if user has required tier for a feature"""
        current_tier = self.get_current_tier()
        
        tier_levels = {
            UserTier.FREE: 0,
            UserTier.PROFESSIONAL: 1,
            UserTier.ENTERPRISE: 2,
            UserTier.ENTERPRISE_PLUS: 3
        }
        
        if tier_levels.get(current_tier, 0) >= tier_levels.get(required_tier, 999):
            return True
        
        # Show upgrade message
        required_tier_info = UserTier.TIER_FEATURES[required_tier]
        st.warning(f"⬆️ **Upgrade Required**")
        st.info(f"""
        **{feature_name}** requires **{required_tier_info['name']}** tier or higher.
        
        **Your tier:** {UserTier.TIER_FEATURES[current_tier]['name']}
        
        Contact sales@encinitas.ai to upgrade.
        """)
        
        return False


# Global instance
auth_manager = AuthManager()


def require_auth(func):
    """Decorator to require authentication for a function"""
    def wrapper(*args, **kwargs):
        if not auth_manager.is_authenticated():
            st.warning("🔒 Please login to access this feature")
            st.info("Use the sidebar to login with a demo account")
            return None
        return func(*args, **kwargs)
    return wrapper


def require_tier(tier: str, feature_name: str):
    """Decorator to require specific tier for a function"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not auth_manager.require_tier(tier, feature_name):
                return None
            return func(*args, **kwargs)
        return wrapper
    return decorator
