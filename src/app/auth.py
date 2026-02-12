"""
User Authentication and Tier Management for Encinitas
"""

import streamlit as st
import hashlib
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import streamlit.components.v1 as components


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
            st.session_state.monthly_usage = {
                'predictions': 0,
                'last_reset': datetime.now().isoformat()
            }

    def _load_users(self) -> Dict:
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
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)

    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    def authenticate(self, username: str, password: str) -> bool:
        if username not in self.users:
            return False
        if self.users[username]['password_hash'] == self._hash_password(password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.tier = self.users[username]['tier']
            self._load_usage(username)
            return True
        return False

    def logout(self):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.tier = UserTier.FREE

    def is_authenticated(self) -> bool:
        return st.session_state.get('authenticated', False)

    def get_current_tier(self) -> str:
        return st.session_state.get('tier', UserTier.FREE)

    def get_tier_features(self, tier: Optional[str] = None) -> Dict:
        tier = tier or self.get_current_tier()
        return UserTier.TIER_FEATURES.get(tier, UserTier.TIER_FEATURES[UserTier.FREE])

    def _load_usage(self, username: str):
        if 'monthly_usage' in self.users[username]:
            usage = self.users[username]['monthly_usage']
            last_reset = datetime.fromisoformat(usage['last_reset'])
            if datetime.now().month != last_reset.month:
                usage = {'predictions': 0, 'last_reset': datetime.now().isoformat()}
                self.users[username]['monthly_usage'] = usage
                self._save_users()
            st.session_state.monthly_usage = usage
        else:
            st.session_state.monthly_usage = {
                'predictions': 0, 'last_reset': datetime.now().isoformat()
            }

    def increment_usage(self, feature: str):
        if not self.is_authenticated():
            return
        username = st.session_state.username
        if feature == 'predictions':
            st.session_state.monthly_usage['predictions'] += 1
            if username in self.users:
                self.users[username]['monthly_usage'] = st.session_state.monthly_usage
                self._save_users()

    def check_usage_limit(self, feature: str) -> bool:
        limit = self.get_tier_features().get(f'{feature}_per_month', -1)
        if limit == -1:
            return True
        return st.session_state.monthly_usage.get(feature, 0) < limit

    def render_login_page(self):
        """
        Render the Encinitas login page.

        Uses st.components.v1.html() to bypass Streamlit's HTML sanitizer
        so we can render a real <form> with full styling.

        Flow:
          1. Render full-page HTML login form via components.html()
          2. Form submits via GET to the same URL → ?_u=…&_p=…&_login=1
          3. Streamlit re-renders, this function reads query_params
          4. If credentials valid → authenticate() → st.rerun()
          5. If invalid → show error and re-render form
        """
        # ── Step 1: handle submitted credentials ─────────────────────────
        params = st.query_params
        login_attempted = params.get("_login") == "1"
        login_error = False

        if login_attempted:
            u = params.get("_u", "")
            p = params.get("_p", "")
            st.query_params.clear()          # prevent re-submit on refresh
            if self.authenticate(u, p):
                st.rerun()
                return
            else:
                login_error = True

        # ── Step 2: load logo ─────────────────────────────────────────────
        logo_src = ""
        # Try SVG first, then PNG files
        for logo_name in ("encinitalogo.svg", "encinitalogo.png", "encinitaslogo.png"):
            logo_path = Path(__file__).parent / logo_name
            if logo_path.exists():
                raw = logo_path.read_bytes()
                # Determine MIME type based on file extension
                mime_type = "image/svg+xml" if logo_name.endswith('.svg') else "image/png"
                logo_src = f"data:{mime_type};base64,{base64.b64encode(raw).decode()}"
                break

        logo_html = (
            f'<img src="{logo_src}" alt="Encinitas" '
            f'style="width:200px;height:auto;display:block;margin:0 auto 6px;">'
            if logo_src else
            '<p style="font-size:1.6rem;font-weight:800;color:#545454;'
            'text-align:center;margin:0 0 6px;">Encinitas</p>'
        )

        error_html = (
            '<div style="color:#dc2626;background:#fef2f2;border:1px solid #fecaca;'
            'border-radius:8px;padding:9px 13px;font-size:13px;margin-bottom:14px;">'
            '&#10060; Invalid username or password</div>'
            if login_error else ""
        )

        # ── Step 3: render via components.html (bypasses sanitizer) ───────
        # The form uses GET so the credentials appear in query_params on submit,
        # which Streamlit picks up on the very next render cycle.
        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, 'Inter', sans-serif; }}
  html, body {{
    height: 100%;
    background: linear-gradient(135deg, #f0fffe 0%, #f8ffff 50%, #f0f9ff 100%);
  }}
  body {{
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    min-height: 540px;
  }}
  .card {{
    background: white;
    border-radius: 20px;
    box-shadow: 0 8px 48px rgba(18,219,180,0.14), 0 2px 12px rgba(0,0,0,0.07);
    padding: 44px 40px 36px;
    width: 100%;
    max-width: 400px;
  }}
  .logo-wrap {{ text-align: center; margin-bottom: 8px; }}
  .tagline {{
    text-align: center; color: #12dbb4; font-weight: 500;
    font-size: 14px; letter-spacing: 0.05em; margin-bottom: 20px;
  }}
  hr {{ border: none; border-top: 1px solid #e8f8f5; margin-bottom: 20px; }}
  label {{
    display: block; font-size: 13px; font-weight: 600;
    color: #545454; margin-bottom: 5px;
  }}
  input[type=text], input[type=password] {{
    width: 100%; border: 1.5px solid #e2e8f0; border-radius: 10px;
    padding: 10px 14px; font-size: 15px; outline: none;
    margin-bottom: 14px; transition: border-color .2s, box-shadow .2s;
  }}
  input[type=text]:focus, input[type=password]:focus {{
    border-color: #12dbb4;
    box-shadow: 0 0 0 3px rgba(18,219,180,0.15);
  }}
  button {{
    width: 100%;
    background: linear-gradient(90deg, #12dbb4, #14d8e2);
    color: white; font-weight: 600; font-size: 15px;
    border: none; border-radius: 10px; padding: 12px;
    cursor: pointer; margin-top: 4px; transition: opacity .2s;
  }}
  button:hover {{ opacity: 0.88; }}
  details {{ margin-top: 16px; }}
  summary {{
    cursor: pointer; font-size: 12px; color: #64748b;
    user-select: none; list-style: none;
  }}
  summary::marker {{ display: none; }}
  summary::before {{ content: "▸  "; }}
  details[open] summary::before {{ content: "▾  "; }}
  .demo-box {{
    font-size: 12px; color: #475569; margin-top: 8px;
    background: #f8fafc; border-radius: 8px;
    padding: 10px 12px; line-height: 1.9;
  }}
  .demo-box code {{
    background: #e2e8f0; border-radius: 4px;
    padding: 1px 5px; font-family: monospace; font-size: 11px;
  }}
  .footer {{
    text-align: center; font-size: 11px; color: #94a3b8;
    margin-top: 16px;
  }}
  .footer a {{ color: #12dbb4; text-decoration: none; }}
</style>
</head>
<body>
<div class="card">
  <div class="logo-wrap">{logo_html}</div>
  <p class="tagline">Clinical Trial Intelligence</p>
  <hr>

  {error_html}

  <form method="GET" action="">
    <input type="hidden" name="_login" value="1">
    <label>Username</label>
    <input type="text" name="_u" placeholder="Enter your username" autocomplete="username">
    <label>Password</label>
    <input type="password" name="_p" placeholder="Enter your password" autocomplete="current-password">
    <button type="submit">Sign In</button>
  </form>

  <details>
    <summary>Demo accounts</summary>
    <div class="demo-box">
      <strong>Free:</strong> <code>demo</code> / <code>demo123</code><br>
      <strong>Professional:</strong> <code>pro_demo</code> / <code>pro123</code><br>
      <strong>Enterprise:</strong> <code>enterprise_demo</code> / <code>ent123</code>
    </div>
  </details>

  <p class="footer">
    &copy; 2026 Encinitas &nbsp;&middot;&nbsp;
    <a href="?legal_page=terms" target="_parent">Terms</a>
    &nbsp;&middot;&nbsp;
    <a href="?legal_page=privacy" target="_parent">Privacy</a>
  </p>
</div>
</body>
</html>"""

        # Hide Streamlit chrome on the login page
        st.markdown("""<style>
[data-testid="stAppViewContainer"] > section:first-child,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }
.block-container { padding: 0 !important; }
iframe { display: block; }
</style>""", unsafe_allow_html=True)

        components.html(html, height=580, scrolling=False)

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
