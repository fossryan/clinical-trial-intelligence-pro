"""
User Authentication and Tier Management for Encinitas
"""

import streamlit as st
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class UserTier:
    """Subscription tiers"""
    
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
            'protocol_optimizer': False,
            'financial_calculator': True,
            'export_excel': False,
        },
        PROFESSIONAL: {
            'name': 'Professional',
            'price': 25000,
            'predictions_per_month': -1,
            'competitor_tracking': 3,
            'protocol_optimizer': False,
            'financial_calculator': True,
            'export_excel': True,
        },
        ENTERPRISE: {
            'name': 'Enterprise',
            'price': 75000,
            'predictions_per_month': -1,
            'competitor_tracking': 10,
            'protocol_optimizer': True,
            'financial_calculator': True,
            'export_excel': True,
            'export_powerpoint': True,
        },
        ENTERPRISE_PLUS: {
            'name': 'Enterprise+',
            'price': 150000,
            'predictions_per_month': -1,
            'competitor_tracking': -1,
            'protocol_optimizer': True,
            'financial_calculator': True,
            'export_excel': True,
            'export_powerpoint': True,
            'custom_model_training': True,
        }
    }


class AuthManager:
    """Manages user authentication"""
    
    def __init__(self, users_file: str = "data/users.json"):
        self.users_file = Path(users_file)
        self.users = self._load_users()
        
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
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.tier = self.users[username]['tier']
            self._load_usage(username)
            return True
        
        return False
    
    def logout(self):
        """Logout current user"""
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.tier = UserTier.FREE
    
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
    
    def _load_usage(self, username: str):
        """Load usage data for user"""
        if 'monthly_usage' in self.users[username]:
            usage = self.users[username]['monthly_usage']
            last_reset = datetime.fromisoformat(usage['last_reset'])
            if datetime.now().month != last_reset.month:
                usage = {'predictions': 0, 'last_reset': datetime.now().isoformat()}
                self.users[username]['monthly_usage'] = usage
                self._save_users()
            st.session_state.monthly_usage = usage
        else:
            st.session_state.monthly_usage = {'predictions': 0, 'last_reset': datetime.now().isoformat()}
    
    def increment_usage(self, feature: str):
        """Increment usage counter"""
        if not self.is_authenticated():
            return
        
        username = st.session_state.username
        
        if feature == 'predictions':
            st.session_state.monthly_usage['predictions'] += 1
            if username in self.users:
                self.users[username]['monthly_usage'] = st.session_state.monthly_usage
                self._save_users()
    
    def check_usage_limit(self, feature: str) -> bool:
        """Check if user has exceeded usage limit"""
        limit = self.get_tier_features().get(f'{feature}_per_month', -1)
        
        if limit == -1:
            return True
        
        current_usage = st.session_state.monthly_usage.get(feature, 0)
        return current_usage < limit
    
    def render_login_page(self):
        """Render Encinitas login page with branding"""
        st.markdown("""
        <style>
            /* Login page styling matching Encinitas brand */
            .login-container {
                max-width: 500px;
                margin: 100px auto;
                padding: 40px;
                background: white;
                border-radius: 20px;
                box-shadow: 0 10px 40px rgba(18, 219, 180, 0.1);
            }
            .login-logo {
                text-align: center;
                margin-bottom: 30px;
            }
            .login-title {
                font-size: 2rem;
                font-weight: 700;
                color: #545454;
                text-align: center;
                margin-bottom: 10px;
            }
            .login-subtitle {
                text-align: center;
                color: #12dbb4;
                margin-bottom: 30px;
            }
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            
            # Logo
            st.markdown('<div class="login-logo">🧬</div>', unsafe_allow_html=True)
            
            # Title
            st.markdown('<div class="login-title">Encinitas</div>', unsafe_allow_html=True)
            st.markdown('<div class="login-subtitle">Clinical Trial Intelligence</div>', unsafe_allow_html=True)
            
            # Demo accounts
            with st.expander("📋 Demo Accounts"):
                st.markdown("""
                **Free Tier:** `demo` / `demo123`  
                **Professional:** `pro_demo` / `pro123`  
                **Enterprise:** `enterprise_demo` / `ent123`
                """)
            
            st.markdown("---")
            
            # Login form
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                
                login_button = st.form_submit_button("🔓 Sign In", use_container_width=True)
                
                if login_button:
                    if self.authenticate(username, password):
                        st.success(f"✅ Welcome, {username}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    def require_tier(self, required_tier: str, feature_name: str) -> bool:
        """Check if user has required tier"""
        current_tier = self.get_current_tier()
        
        tier_levels = {
            UserTier.FREE: 0,
            UserTier.PROFESSIONAL: 1,
            UserTier.ENTERPRISE: 2,
            UserTier.ENTERPRISE_PLUS: 3
        }
        
        if tier_levels.get(current_tier, 0) >= tier_levels.get(required_tier, 999):
            return True
        
        required_tier_info = UserTier.TIER_FEATURES[required_tier]
        st.warning(f"⬆️ Upgrade to {required_tier_info['name']} Required")
        st.info(f"""
        **{feature_name}** requires **{required_tier_info['name']}** tier.
        
        Your current tier: **{UserTier.TIER_FEATURES[current_tier]['name']}**
        
        Contact sales@encinitas.ai to upgrade.
        """)
        
        return False


# Global instance
auth_manager = AuthManager()
