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
        """Render Encinitas login page — everything in one st.markdown call."""
        import base64, hashlib

        # ── Logo ─────────────────────────────────────────────────────────
        logo_path = Path(__file__).parent / "encinitalogo.png"
        logo_src = ""
        if logo_path.exists():
            with open(logo_path, "rb") as fh:
                logo_src = "data:image/png;base64," + base64.b64encode(fh.read()).decode()

        # ── Handle form submission via query params ───────────────────────
        # The HTML form POSTs to Streamlit by setting ?_login=1&u=...&p=...
        # We read those params here and authenticate before rendering.
        params = st.query_params
        if params.get("_login") == "1":
            u = params.get("u", "")
            p = params.get("p", "")
            # Clear params immediately so refresh doesn't re-submit
            st.query_params.clear()
            if self.authenticate(u, p):
                st.rerun()
            else:
                st.session_state["_login_error"] = True
                st.rerun()

        login_error = st.session_state.pop("_login_error", False)
        error_html = (
            '<p style="color:#dc2626;background:#fef2f2;border:1px solid #fecaca;'
            'border-radius:8px;padding:8px 12px;font-size:.85rem;margin:0 0 10px;">'
            '❌ Invalid username or password</p>'
            if login_error else ""
        )

        logo_img = (
            f'<img src="{logo_src}" style="width:200px;height:auto;display:block;margin:0 auto 8px;" alt="Encinitas">'
            if logo_src else
            '<p style="text-align:center;font-size:1.5rem;font-weight:700;color:#545454;">Encinitas</p>'
        )

        # ── Single st.markdown — CSS + logo + form, zero ghost elements ──
        # Using a real HTML <form> that navigates to ?_login=1&u=…&p=…
        # Streamlit re-renders, reads params, authenticates, then st.rerun().
        st.markdown(f"""
<style>
[data-testid="stAppViewContainer"]{{background:linear-gradient(135deg,#f0fffe,#f8ffff,#f0f9ff)!important;}}
[data-testid="stHeader"],[data-testid="stToolbar"]{{background:transparent!important;}}
[data-testid="stSidebar"]{{display:none!important;}}
.block-container{{padding-top:0!important;max-width:100%!important;}}
section.main .block-container{{padding:0!important;}}
</style>
<div style="min-height:100vh;display:flex;align-items:center;justify-content:center;
            background:linear-gradient(135deg,#f0fffe,#f8ffff,#f0f9ff);padding:24px;">
  <div style="background:white;border-radius:20px;
              box-shadow:0 8px 48px rgba(18,219,180,.14),0 2px 12px rgba(0,0,0,.06);
              padding:44px 40px 36px;width:100%;max-width:420px;">

    <div style="text-align:center;margin-bottom:6px;">{logo_img}</div>
    <p style="text-align:center;color:#12dbb4;font-weight:500;font-size:.9rem;
              letter-spacing:.05em;margin:0 0 20px;">Clinical Trial Intelligence</p>
    <hr style="border:none;border-top:1px solid #e8f8f5;margin:0 0 20px;">

    {error_html}

    <form method="GET" action="" style="margin:0;">
      <input type="hidden" name="_login" value="1">

      <label style="display:block;font-size:.83rem;font-weight:600;color:#545454;margin-bottom:5px;">
        Username
      </label>
      <input name="u" type="text" placeholder="Enter your username"
        style="width:100%;box-sizing:border-box;border-radius:10px;
               border:1.5px solid #e2e8f0;padding:10px 14px;font-size:.95rem;
               margin-bottom:14px;outline:none;font-family:inherit;"
        onfocus="this.style.borderColor='#12dbb4';this.style.boxShadow='0 0 0 3px rgba(18,219,180,.15)'"
        onblur="this.style.borderColor='#e2e8f0';this.style.boxShadow='none'">

      <label style="display:block;font-size:.83rem;font-weight:600;color:#545454;margin-bottom:5px;">
        Password
      </label>
      <input name="p" type="password" placeholder="Enter your password"
        style="width:100%;box-sizing:border-box;border-radius:10px;
               border:1.5px solid #e2e8f0;padding:10px 14px;font-size:.95rem;
               margin-bottom:18px;outline:none;font-family:inherit;"
        onfocus="this.style.borderColor='#12dbb4';this.style.boxShadow='0 0 0 3px rgba(18,219,180,.15)'"
        onblur="this.style.borderColor='#e2e8f0';this.style.boxShadow='none'">

      <button type="submit"
        style="width:100%;background:linear-gradient(90deg,#12dbb4,#14d8e2);
               color:white;font-weight:600;font-size:1rem;border:none;
               border-radius:10px;padding:12px;cursor:pointer;font-family:inherit;"
        onmouseover="this.style.opacity='.88'" onmouseout="this.style.opacity='1'">
        Sign In
      </button>
    </form>

    <details style="margin-top:16px;">
      <summary style="cursor:pointer;font-size:.82rem;color:#64748b;user-select:none;">
        Demo accounts
      </summary>
      <div style="font-size:.82rem;color:#475569;margin-top:8px;
                  background:#f8fafc;border-radius:8px;padding:10px 12px;line-height:1.8;">
        <strong>Free:</strong> demo / demo123<br>
        <strong>Professional:</strong> pro_demo / pro123<br>
        <strong>Enterprise:</strong> enterprise_demo / ent123
      </div>
    </details>

    <p style="text-align:center;font-size:.74rem;color:#94a3b8;margin-top:14px;margin-bottom:0;">
      © 2026 Encinitas &nbsp;·&nbsp;
      <a href="?legal_page=terms" style="color:#12dbb4;text-decoration:none;">Terms</a>
      &nbsp;·&nbsp;
      <a href="?legal_page=privacy" style="color:#12dbb4;text-decoration:none;">Privacy</a>
    </p>
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
