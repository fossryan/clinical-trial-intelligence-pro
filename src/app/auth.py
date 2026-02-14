"""
User Authentication and Tier Management for Encinitas
Updated with improved login design and collapsible demo accounts
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
        if 'show_demo_accounts' not in st.session_state:
            st.session_state.show_demo_accounts = False
    
    def _load_users(self) -> Dict:
        """Load users from JSON file"""
        if not self.users_file.exists():
            # Enhanced demo accounts with better credentials
            default_users = {
                'demo': {
                    'password_hash': self._hash_password('demo123'),
                    'tier': UserTier.FREE,
                    'email': 'demo@encinitas.ai',
                    'role': 'Free User',
                    'created_at': datetime.now().isoformat()
                },
                'pro_demo': {
                    'password_hash': self._hash_password('pro123'),
                    'tier': UserTier.PROFESSIONAL,
                    'email': 'pro@encinitas.ai',
                    'role': 'Professional User',
                    'created_at': datetime.now().isoformat()
                },
                'enterprise_demo': {
                    'password_hash': self._hash_password('ent123'),
                    'tier': UserTier.ENTERPRISE,
                    'email': 'enterprise@encinitas.ai',
                    'role': 'Enterprise User',
                    'created_at': datetime.now().isoformat()
                },
                # Additional professional demo accounts
                'pharma.exec': {
                    'password_hash': self._hash_password('demo2024'),
                    'tier': UserTier.ENTERPRISE,
                    'email': 'pharma.exec@bigtrial.com',
                    'role': 'Pharma Executive',
                    'created_at': datetime.now().isoformat()
                },
                'biotech.founder': {
                    'password_hash': self._hash_password('demo2024'),
                    'tier': UserTier.PROFESSIONAL,
                    'email': 'founder@biotech-startup.com',
                    'role': 'Biotech Founder',
                    'created_at': datetime.now().isoformat()
                },
                'investor': {
                    'password_hash': self._hash_password('demo2024'),
                    'tier': UserTier.PROFESSIONAL,
                    'email': 'investor@healthcare-vc.com',
                    'role': 'Healthcare Investor',
                    'created_at': datetime.now().isoformat()
                },
                'researcher': {
                    'password_hash': self._hash_password('demo2024'),
                    'tier': UserTier.FREE,
                    'email': 'researcher@university.edu',
                    'role': 'Clinical Researcher',
                    'created_at': datetime.now().isoformat()
                },
                'analyst': {
                    'password_hash': self._hash_password('demo2024'),
                    'tier': UserTier.PROFESSIONAL,
                    'email': 'analyst@consulting-firm.com',
                    'role': 'Data Analyst',
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
            st.session_state.user_email = self.users[username].get('email', '')
            st.session_state.user_role = self.users[username].get('role', 'User')
            self._load_usage(username)
            return True
        
        return False
    
    def logout(self):
        """Logout current user"""
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.tier = UserTier.FREE
        st.session_state.user_email = None
        st.session_state.user_role = None
    
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
        """Render Encinitas login page with SVG logo and improved demo accounts section"""
        
        # Handle signup success message with custom styling
        show_success = False
        signup_params = st.query_params
        if signup_params.get("_signup_success") == "1":
            show_success = True

        # Inline SVG logo
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

        # Modern styling with improved demo accounts section
        st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

            /* Hide Streamlit's default success message dismiss button */
            [data-testid="stNotification"] button {{
                display: none !important;
            }}
            
            /* Custom success message */
            .custom-success-message {{
                position: relative;
                background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
                border: 1px solid #a7f3d0;
                border-radius: 12px;
                padding: 14px 40px 14px 16px;
                margin-bottom: 20px;
                color: #059669;
                font-family: 'Inter', sans-serif;
                font-size: 14px;
                font-weight: 500;
                box-shadow: 0 2px 8px rgba(5, 150, 105, 0.1);
            }}
            
            .custom-success-message .close-btn {{
                position: absolute;
                top: 12px;
                right: 12px;
                background: none;
                border: none;
                color: #059669;
                font-size: 20px;
                cursor: pointer;
                padding: 4px 8px;
                line-height: 1;
                transition: opacity 0.2s;
            }}
            
            .custom-success-message .close-btn:hover {{
                opacity: 0.7;
            }}

            /* Page background */
            [data-testid="stAppViewContainer"] {{
                background: linear-gradient(135deg, #e8fcf8 0%, #e0f7fc 50%, #dbeef9 100%) !important;
            }}
            [data-testid="stHeader"] {{ background: transparent !important; }}

            /* Centre the column and add card feel */
            .block-container {{
                padding-top: 4vh !important;
                max-width: 100% !important;
                padding-bottom: 2vh !important;
            }}

            /* The middle column becomes the modern card */
            div[data-testid="column"]:nth-child(2) > div:first-child {{
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(20px);
                border-radius: 24px;
                padding: 48px 44px 32px !important;
                box-shadow: 0 20px 60px rgba(18, 219, 180, 0.08), 
                            0 4px 16px rgba(0, 0, 0, 0.04),
                            inset 0 1px 0 rgba(255, 255, 255, 0.8);
                border: 1px solid rgba(18, 219, 180, 0.1);
            }}

            /* Logo block */
            .enc-logo-block {{
                text-align: center;
                margin-bottom: 12px;
            }}

            /* Welcome text */
            .enc-welcome h1 {{
                font-family: 'Inter', sans-serif;
                font-weight: 700;
                font-size: 32px;
                margin: 0 0 6px 0;
                text-align: center;
                background: linear-gradient(135deg, #0f766e 0%, #12dbb4 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}

            /* Tagline */
            .enc-tagline {{
                font-family: 'Inter', sans-serif;
                color: #64748b;
                font-size: 15px;
                text-align: center;
                margin-bottom: 32px;
                font-weight: 500;
            }}

            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 8px;
                background: rgba(18, 219, 180, 0.05);
                padding: 6px;
                border-radius: 12px;
            }}

            .stTabs [data-baseweb="tab"] {{
                font-family: 'Inter', sans-serif;
                font-weight: 600;
                font-size: 15px;
                padding: 10px 20px;
                border-radius: 8px;
                transition: all 0.2s;
                color: #64748b;
            }}

            .stTabs [aria-selected="true"] {{
                background: linear-gradient(135deg, #12dbb4 0%, #14d8e2 100%) !important;
                color: white !important;
                box-shadow: 0 4px 12px rgba(18, 219, 180, 0.25);
            }}

            /* Form inputs */
            .stTextInput > div > div > input {{
                font-family: 'Inter', sans-serif;
                font-size: 15px;
                border-radius: 10px;
                border: 2px solid #e2e8f0;
                padding: 12px 16px;
                transition: all 0.2s;
                background: #f8fafc;
            }}

            .stTextInput > div > div > input:focus {{
                border-color: #12dbb4;
                background: white;
                box-shadow: 0 0 0 3px rgba(18, 219, 180, 0.1);
            }}

            /* Submit buttons */
            .stButton > button {{
                font-family: 'Inter', sans-serif;
                font-weight: 600;
                font-size: 15px;
                padding: 12px 24px;
                border-radius: 10px;
                background: linear-gradient(135deg, #12dbb4 0%, #14d8e2 100%);
                border: none;
                color: white;
                transition: all 0.3s;
                box-shadow: 0 4px 12px rgba(18, 219, 180, 0.25);
            }}

            .stButton > button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(18, 219, 180, 0.35);
            }}

            /* Expander for demo accounts */
            .stExpander {{
                background: linear-gradient(135deg, #f0fdfa 0%, #ecfdf5 100%);
                border: 2px solid #14B8A6;
                border-radius: 12px;
                margin-top: 16px;
            }}

            .stExpander [data-testid="stExpanderDetails"] {{
                padding: 16px;
            }}

            /* Demo account cards */
            .demo-account-card {{
                background: white;
                border: 1px solid #99f6e4;
                border-radius: 8px;
                padding: 12px 16px;
                margin: 8px 0;
                transition: all 0.2s;
                cursor: pointer;
            }}

            .demo-account-card:hover {{
                border-color: #14B8A6;
                box-shadow: 0 2px 8px rgba(20, 184, 166, 0.15);
                transform: translateX(3px);
            }}

            .demo-role {{
                font-weight: 600;
                color: #0f766e;
                font-size: 14px;
                margin-bottom: 4px;
            }}

            .demo-credentials {{
                font-size: 13px;
                color: #64748B;
                font-family: 'Courier New', monospace;
            }}

            .demo-tier {{
                display: inline-block;
                background: linear-gradient(135deg, #14B8A6 0%, #0D9488 100%);
                color: white;
                padding: 2px 10px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: 600;
                margin-top: 4px;
            }}

            /* Footer */
            .enc-login-footer {{
                text-align: center;
                margin-top: 32px;
                padding-top: 24px;
                border-top: 1px solid #e2e8f0;
                color: #94a3b8;
                font-size: 13px;
                font-family: 'Inter', sans-serif;
            }}

            .enc-login-footer a {{
                color: #12dbb4;
                text-decoration: none;
                transition: color 0.2s;
            }}

            .enc-login-footer a:hover {{
                color: #0fc9a7;
            }}

            /* Features footer */
            .enc-features-footer {{
                max-width: 1200px;
                margin: 48px auto 24px;
                padding: 40px 32px;
                background: rgba(255, 255, 255, 0.7);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                border: 1px solid rgba(18, 219, 180, 0.1);
            }}

            .enc-features-footer h3 {{
                font-family: 'Inter', sans-serif;
                text-align: center;
                color: #0f766e;
                margin-bottom: 32px;
                font-size: 24px;
            }}

            .feature-links {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 32px;
                margin-bottom: 24px;
            }}

            .feature-column {{
                display: flex;
                flex-direction: column;
                gap: 8px;
            }}
            
            .feature-column a {{
                color: #64748b;
                text-decoration: none;
                font-size: 13px;
                transition: all 0.2s;
                padding: 4px 0;
            }}
            
            .feature-column a:hover {{
                color: #12dbb4;
                padding-left: 4px;
            }}
            
            .footer-brand {{
                text-align: center;
                padding-top: 24px;
                border-top: 1px solid #e2e8f0;
                color: #94a3b8;
                font-size: 12px;
            }}
            
            .footer-brand a {{
                color: #12dbb4;
                text-decoration: none;
            }}
            
            .footer-brand a:hover {{
                color: #0fc9a7;
            }}
        </style>
        
        <script>
        function closeSuccessMessage() {{
            const msg = document.getElementById('success-message');
            if (msg) {{
                msg.style.display = 'none';
            }}
        }}
        </script>
        """, unsafe_allow_html=True)

        _, col, _ = st.columns([1, 1.4, 1])

        with col:
            # Custom success message with close button
            if show_success:
                st.markdown("""
                <div id="success-message" class="custom-success-message">
                    <button class="close-btn" onclick="closeSuccessMessage()">×</button>
                    ✅ Account created successfully! Please sign in.
                </div>
                """, unsafe_allow_html=True)
            
            # Logo + tagline
            st.markdown(
                f'<div class="enc-logo-block">{SVG_LOGO}</div>'
                f'<div class="enc-welcome"><h1>Welcome Back</h1></div>'
                f'<p class="enc-tagline">Clinical Trial Intelligence Platform</p>',
                unsafe_allow_html=True,
            )

            # Tabs for Sign In / Sign Up
            tab1, tab2 = st.tabs(["Sign In", "Create Account"])
            
            with tab1:
                with st.form("login_form"):
                    username = st.text_input("Username or Email", placeholder="Enter your username", key="login_username")
                    password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")
                    submitted = st.form_submit_button("Sign In →", use_container_width=True)
                    if submitted:
                        if self.authenticate(username, password):
                            st.success(f"✅ Welcome back, {self.users[username].get('role', 'User')}!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password")
            
            with tab2:
                with st.form("signup_form"):
                    new_email = st.text_input("Email", placeholder="your@email.com", key="signup_email")
                    new_username = st.text_input("Username", placeholder="Choose a username", key="signup_username")
                    new_password = st.text_input("Password", type="password", placeholder="Create a password (min 6 characters)", key="signup_password")
                    signup_submitted = st.form_submit_button("Create Account →", use_container_width=True)
                    
                    if signup_submitted:
                        # Validate inputs
                        if not new_email or not new_username or not new_password:
                            st.error("❌ All fields are required")
                        elif new_username in self.users:
                            st.error("❌ Username already exists")
                        elif len(new_password) < 6:
                            st.error("❌ Password must be at least 6 characters")
                        elif "@" not in new_email or "." not in new_email:
                            st.error("❌ Please enter a valid email address")
                        else:
                            # Create new user
                            self.users[new_username] = {
                                'password_hash': self._hash_password(new_password),
                                'tier': UserTier.FREE,
                                'email': new_email,
                                'role': 'User',
                                'created_at': datetime.now().isoformat()
                            }
                            self._save_users()
                            st.query_params["_signup_success"] = "1"
                            st.rerun()

            # Improved demo accounts section - Collapsible
            with st.expander("🎯 Try Demo Access", expanded=False):
                st.markdown("**Click any account to see credentials:**")
                
                # Group demo accounts by tier
                demo_accounts_display = {
                    'Enterprise Tier': [
                        ('pharma.exec', '🏢 Pharma Executive', 'pharma.exec@bigtrial.com'),
                        ('enterprise_demo', '🏢 Enterprise Demo', 'enterprise@encinitas.ai'),
                    ],
                    'Professional Tier': [
                        ('biotech.founder', '🧬 Biotech Founder', 'founder@biotech-startup.com'),
                        ('investor', '💰 Healthcare Investor', 'investor@healthcare-vc.com'),
                        ('analyst', '📊 Data Analyst', 'analyst@consulting-firm.com'),
                        ('pro_demo', '💼 Professional Demo', 'pro@encinitas.ai'),
                    ],
                    'Free Tier': [
                        ('researcher', '🔬 Clinical Researcher', 'researcher@university.edu'),
                        ('demo', '🆓 Free Demo', 'demo@encinitas.ai'),
                    ]
                }
                
                for tier_name, accounts in demo_accounts_display.items():
                    st.markdown(f"**{tier_name}:**")
                    for username, display_name, email in accounts:
                        if username in self.users:
                            user_data = self.users[username]
                            tier_info = UserTier.TIER_FEATURES[user_data['tier']]
                            
                            # Determine password (most use demo2024, some use specific passwords)
                            if username in ['demo', 'pro_demo', 'enterprise_demo']:
                                password_hint = {
                                    'demo': 'demo123',
                                    'pro_demo': 'pro123',
                                    'enterprise_demo': 'ent123'
                                }[username]
                            else:
                                password_hint = 'demo2024'
                            
                            st.markdown(f"""
                            <div class="demo-account-card">
                                <div class="demo-role">{display_name}</div>
                                <div class="demo-credentials">
                                    👤 Username: <strong>{username}</strong><br>
                                    🔑 Password: <strong>{password_hint}</strong><br>
                                    📧 Email: {email}
                                </div>
                                <span class="demo-tier">{tier_info['name']} Access</span>
                            </div>
                            """, unsafe_allow_html=True)
                    st.markdown("")

            st.markdown(
                '<div class="enc-login-footer">© 2026 Encinitas &nbsp;&nbsp;·&nbsp;&nbsp; '
                '<a href="?legal_page=terms">Terms</a> &nbsp;&nbsp;·&nbsp;&nbsp; '
                '<a href="?legal_page=privacy">Privacy</a></div>',
                unsafe_allow_html=True,
            )
        
        # Comprehensive features footer outside the login card
        st.markdown("""
        <div class="enc-features-footer">
            <h3>🧬 Platform Features</h3>
            <div class="feature-links">
                <div class="feature-column">
                    <strong style="color: #12dbb4; font-size: 14px; margin-bottom: 4px;">Core Features</strong>
                    <a href="#prediction">📊 Trial Risk Prediction</a>
                    <a href="#batch">📁 Batch Analysis</a>
                    <a href="#portfolio">📂 Portfolio Analyzer</a>
                    <a href="#benchmark">📈 Benchmark Comparisons</a>
                </div>
                <div class="feature-column">
                    <strong style="color: #12dbb4; font-size: 14px; margin-bottom: 4px;">Professional Tools</strong>
                    <a href="#competitive">🎯 Competitive Intelligence</a>
                    <a href="#financial">💰 Financial Calculator</a>
                    <a href="#monitoring">📡 Real-Time Monitoring</a>
                    <a href="#export">📤 Data Export Center</a>
                </div>
                <div class="feature-column">
                    <strong style="color: #12dbb4; font-size: 14px; margin-bottom: 4px;">Enterprise Features</strong>
                    <a href="#protocol">🔬 Protocol Optimizer</a>
                    <a href="#site">🏥 Site Intelligence</a>
                    <a href="#regulatory">📋 Regulatory Advisor</a>
                    <a href="#indication">💊 Indication Recommender</a>
                </div>
                <div class="feature-column">
                    <strong style="color: #12dbb4; font-size: 14px; margin-bottom: 4px;">Resources</strong>
                    <a href="#pricing">💎 Pricing & Plans</a>
                    <a href="#docs">📚 Documentation</a>
                    <a href="mailto:sales@encinitas.ai">📧 Contact Sales</a>
                    <a href="mailto:support@encinitas.ai">🛟 Support</a>
                </div>
            </div>
            <div class="footer-brand">
                <strong style="background: linear-gradient(90deg, #12dbb4, #14d8e2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">ENCINITAS</strong>
                &nbsp;&nbsp;·&nbsp;&nbsp;
                Powered by AI-driven clinical trial intelligence
                &nbsp;&nbsp;·&nbsp;&nbsp;
                <a href="?legal_page=terms">Terms</a>
                &nbsp;&nbsp;·&nbsp;&nbsp;
                <a href="?legal_page=privacy">Privacy</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
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
