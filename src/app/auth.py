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
        """Render Encinitas login page with SVG logo and brand styling"""

        # Inline SVG logo – avoids any file-path issues
        SVG_LOGO = """<svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg"
            xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 532.01 385.6"
            style="width:220px;height:auto;display:block;margin:0 auto 8px;">
          <defs>
            <linearGradient id="lg1" x1="0" y1="346.12" x2="532.01" y2="346.12" gradientUnits="userSpaceOnUse">
              <stop offset="0" stop-color="#55c1a8"/><stop offset="1" stop-color="#4ec5d4"/>
            </linearGradient>
            <linearGradient id="lg2" x1="164.44" y1="125.79" x2="339.12" y2="125.79"
              gradientTransform="translate(185.2 -134.72) rotate(45)" gradientUnits="userSpaceOnUse">
              <stop offset="0" stop-color="#55c1a8"/><stop offset="1" stop-color="#4ec5d4"/>
            </linearGradient>
          </defs>
          <g>
            <path fill="url(#lg1)" d="M0,375.9v-59.55c0-4.69,3.71-8.4,8.4-8.4h42.32c4.14,0,7.53,3.38,7.53,7.53s-3.38,7.42-7.53,7.42H16.69v15.49h29.12c4.14,0,7.53,3.38,7.53,7.53s-3.38,7.42-7.53,7.42h-29.12v16.03h34.58c4.14,0,7.53,3.38,7.53,7.53s-3.38,7.42-7.53,7.42H8.4c-4.69,0-8.4-3.71-8.4-8.4Z"/>
            <path fill="url(#lg1)" d="M73.07,315.91c0-4.69,3.71-8.4,8.4-8.4h1.75c4.04,0,6.44,1.96,8.73,4.91l32.39,42.54v-39.37c0-4.58,3.71-8.29,8.29-8.29s8.29,3.71,8.29,8.29v60.75c0,4.69-3.71,8.4-8.4,8.4h-.55c-4.04,0-6.44-1.96-8.73-4.91l-33.59-44.06v40.9c0,4.58-3.71,8.29-8.29,8.29s-8.29-3.71-8.29-8.29v-60.75Z"/>
            <path fill="url(#lg1)" d="M155.85,346.34v-.22c0-21.7,16.36-39.48,39.81-39.48,11.45,0,19.2,3.05,25.52,7.74,1.75,1.31,3.27,3.71,3.27,6.65,0,4.58-3.71,8.18-8.29,8.18-2.29,0-3.82-.87-5.02-1.64-4.69-3.49-9.6-5.45-15.6-5.45-12.87,0-22.14,10.69-22.14,23.78v.22c0,13.09,9.05,24,22.14,24,7.09,0,11.78-2.18,16.58-6,1.31-1.09,3.05-1.85,5.02-1.85,4.25,0,7.85,3.49,7.85,7.74,0,2.62-1.31,4.69-2.84,6-6.87,6-14.94,9.6-27.16,9.6-22.47,0-39.16-17.34-39.16-39.26Z"/>
            <path fill="url(#lg1)" d="M240.48,315.69c0-4.69,3.71-8.4,8.4-8.4s8.4,3.71,8.4,8.4v60.86c0,4.69-3.71,8.4-8.4,8.4s-8.4-3.71-8.4-8.4v-60.86Z"/>
            <path fill="url(#lg1)" d="M276.36,315.91c0-4.69,3.71-8.4,8.4-8.4h1.75c4.04,0,6.43,1.96,8.73,4.91l32.39,42.54v-39.37c0-4.58,3.71-8.29,8.29-8.29s8.29,3.71,8.29,8.29v60.75c0,4.69-3.71,8.4-8.4,8.4h-.54c-4.04,0-6.44-1.96-8.73-4.91l-33.59-44.06v40.9c0,4.58-3.71,8.29-8.29,8.29s-8.29-3.71-8.29-8.29v-60.75Z"/>
            <path fill="url(#lg1)" d="M363.29,315.69c0-4.69,3.71-8.4,8.4-8.4s8.4,3.71,8.4,8.4v60.86c0,4.69-3.71,8.4-8.4,8.4s-8.4-3.71-8.4-8.4v-60.86Z"/>
            <path fill="url(#lg1)" d="M416.95,323.43h-16.14c-4.25,0-7.74-3.49-7.74-7.74s3.49-7.74,7.74-7.74h49.08c4.25,0,7.74,3.49,7.74,7.74s-3.49,7.74-7.74,7.74h-16.14v53.12c0,4.69-3.71,8.4-8.4,8.4s-8.4-3.71-8.4-8.4v-53.12Z"/>
            <path fill="url(#lg1)" d="M457.41,373.5l26.61-60.1c1.85-4.14,5.24-6.65,9.82-6.65h.98c4.58,0,7.85,2.51,9.71,6.65l26.61,60.1c.54,1.2.87,2.29.87,3.38,0,4.47-3.49,8.07-7.96,8.07-3.93,0-6.54-2.29-8.07-5.78l-5.13-12h-33.59l-5.34,12.54c-1.42,3.27-4.25,5.24-7.74,5.24-4.36,0-7.74-3.49-7.74-7.85,0-1.2.44-2.4.98-3.6ZM504.63,352.34l-10.58-25.2-10.58,25.2h21.16Z"/>
          </g>
          <g>
            <path fill="url(#lg2)" d="M331.59,162.22c9.94,4.03,12.51,17.84,8.41,26.13-5.17,10.46-17.4,15.4-27.92,10.73-8-3.55-12.54-14.6-9.25-24.03,3.58-10.25,15.3-18.29,28.75-12.83Z"/>
            <path fill="url(#lg2)" d="M250.57,261.5c13.95-3.85,27.46-10.97,39.24-21.57,1.55-1.4,2.7-3.01,3.68-4.7l5.8-6.06c7.96-8.07,8.88-21.15,2.88-28.08-7.57-8.74-21.52-8.47-30.6.56l-9.57,9.52c-7.69,6.37-16.3,10.42-24.98,12.2-14.69,2.72-29.48-1.23-39.9-11.44-15.79-15.49-19.19-43.7-2.11-64.19,9.64-11.57,20.59-21.91,31.46-32.27l46.22,45.6c4.14,4.07,9.98,5.79,15.63,5.47.51-.01,1.01-.05,1.52-.1.45-.06.89-.13,1.34-.21,4.82-.78,9.49-3.13,13.08-7.19,6.55-7.41,6.62-20.45-.74-27.94l-46.31-46.94,28.8-28.5c20.3-20.09,51.89-19.39,68.68-1.59,16.59,17.59,16.3,45.37-1.07,66.36l-9.52,9.57c-9.03,9.08-9.3,23.03-.56,30.6,6.93,6,20,5.08,28.08-2.88l9.37-8.97c.53-.47,1.17-1.19,1.65-1.74,33.31-37.64,33.19-90.04,1.53-122.11-29.51-29.9-82.72-35.37-119.91-2.51l-.2-.24c-35.06,30.33-67.41,63.9-99,98.34-34.16,37.24-28.84,91.38,1.41,121.24,20.66,20.39,49.74,27.61,77.88,21.33.34-.07.67-.17,1.01-.24,1.73-.41,3.47-.79,5.19-1.31Z"/>
          </g>
        </svg>"""

        # ----------------------------------------------------------------
        # All visual chrome is pure CSS applied to Streamlit's own
        # elements — no wrapping <div> that produces a blank element.
        # ----------------------------------------------------------------
        st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

            /* Page background */
            [data-testid="stAppViewContainer"] {{
                background: linear-gradient(135deg,#f0fffe 0%,#f8ffff 50%,#f0f9ff 100%) !important;
            }}
            [data-testid="stHeader"] {{ background: transparent !important; }}

            /* Centre the column and add card feel via the column's own block */
            .block-container {{
                padding-top: 5vh !important;
                max-width: 100% !important;
            }}

            /* The middle column becomes the card */
            div[data-testid="column"]:nth-child(2) > div:first-child {{
                background: white;
                border-radius: 20px;
                padding: 40px 36px 32px !important;
                box-shadow: 0 8px 48px rgba(18,219,180,0.13), 0 2px 12px rgba(0,0,0,0.06);
            }}

            /* Logo block – sits above the form, no extra wrapper needed */
            .enc-logo-block {{
                text-align: center;
                padding-bottom: 6px;
            }}
            .enc-tagline {{
                text-align: center;
                font-size: 0.9rem;
                color: #12dbb4;
                font-weight: 500;
                letter-spacing: 0.05em;
                margin: 0 0 18px;
                font-family: 'Inter', sans-serif;
            }}
            .enc-rule {{
                border: none;
                border-top: 1px solid #e8f8f5;
                margin: 0 0 20px;
            }}

            /* Input fields */
            .stTextInput > label {{
                font-size: 0.82rem !important;
                font-weight: 600 !important;
                color: #545454 !important;
                font-family: 'Inter', sans-serif !important;
            }}
            .stTextInput > div > input {{
                border-radius: 10px !important;
                border: 1.5px solid #e2e8f0 !important;
                padding: 10px 14px !important;
                font-size: 0.95rem !important;
                font-family: 'Inter', sans-serif !important;
            }}
            .stTextInput > div > input:focus {{
                border-color: #12dbb4 !important;
                box-shadow: 0 0 0 3px rgba(18,219,180,0.15) !important;
                outline: none !important;
            }}

            /* Sign-in button */
            .stFormSubmitButton > button {{
                width: 100% !important;
                background: linear-gradient(90deg,#12dbb4 0%,#14d8e2 100%) !important;
                color: white !important;
                font-weight: 600 !important;
                font-size: 1rem !important;
                border: none !important;
                border-radius: 10px !important;
                padding: 12px !important;
                margin-top: 6px !important;
                cursor: pointer !important;
                transition: opacity .2s !important;
                font-family: 'Inter', sans-serif !important;
            }}
            .stFormSubmitButton > button:hover {{ opacity: .88 !important; }}

            /* Footer */
            .enc-login-footer {{
                text-align: center;
                font-size: 0.76rem;
                color: #94a3b8;
                margin-top: 16px;
                font-family: 'Inter', sans-serif;
            }}
            .enc-login-footer a {{
                color: #12dbb4;
                text-decoration: none;
            }}
        </style>
        """, unsafe_allow_html=True)

        _, col, _ = st.columns([1, 1.4, 1])

        with col:
            # Logo + tagline rendered as a single markdown — no wrapping div
            st.markdown(
                f'<div class="enc-logo-block">{SVG_LOGO}</div>'
                f'<p class="enc-tagline">Clinical Trial Intelligence</p>'
                f'<hr class="enc-rule">',
                unsafe_allow_html=True,
            )

            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("Sign In", use_container_width=True)
                if submitted:
                    if self.authenticate(username, password):
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

            with st.expander("Demo accounts"):
                st.markdown(
                    "**Free:** `demo` / `demo123`  \n"
                    "**Professional:** `pro_demo` / `pro123`  \n"
                    "**Enterprise:** `enterprise_demo` / `ent123`"
                )

            st.markdown(
                '<p class="enc-login-footer">© 2026 Encinitas &nbsp;·&nbsp; '
                '<a href="?legal_page=terms">Terms of Service</a> &nbsp;·&nbsp; '
                '<a href="?legal_page=privacy">Privacy Policy</a></p>',
                unsafe_allow_html=True,
            )
    
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
