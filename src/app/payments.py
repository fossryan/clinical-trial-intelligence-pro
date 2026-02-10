"""
Stripe Payment Processing for Encinitas
Handles subscription management for Professional, Enterprise, Enterprise+ tiers
"""

import streamlit as st
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Stripe is optional – app degrades gracefully without it
try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False


# ---------------------------------------------------------------------------
# STRIPE CONFIGURATION
# ---------------------------------------------------------------------------
# Set your keys in .streamlit/secrets.toml:
#
#   [stripe]
#   secret_key      = "sk_live_..."      # or sk_test_... for testing
#   publishable_key = "pk_live_..."
#   webhook_secret  = "whsec_..."
#
#   [stripe.prices]
#   professional  = "price_XXXXXXXXXXXXXXXX"
#   enterprise    = "price_XXXXXXXXXXXXXXXX"
#   enterprise_plus = "price_XXXXXXXXXXXXXXXX"

TIER_PRICES = {
    "professional":   {"name": "Professional",  "amount": 25_000,  "display": "$25,000 / year"},
    "enterprise":     {"name": "Enterprise",     "amount": 75_000,  "display": "$75,000 / year"},
    "enterprise_plus":{"name": "Enterprise+",   "amount": 150_000, "display": "$150,000 / year"},
}


def _get_stripe_keys() -> Dict[str, str]:
    """Read Stripe keys from Streamlit secrets (never hard-code)."""
    try:
        return {
            "secret_key":      st.secrets["stripe"]["secret_key"],
            "publishable_key": st.secrets["stripe"]["publishable_key"],
            "webhook_secret":  st.secrets["stripe"].get("webhook_secret", ""),
        }
    except Exception:
        return {}


def _get_price_id(tier: str) -> Optional[str]:
    """Fetch the Stripe Price ID for a given tier from secrets."""
    try:
        return st.secrets["stripe"]["prices"][tier]
    except Exception:
        return None


def _init_stripe():
    """Initialise the Stripe SDK with the secret key."""
    if not STRIPE_AVAILABLE:
        return False
    keys = _get_stripe_keys()
    if not keys.get("secret_key"):
        return False
    stripe.api_key = keys["secret_key"]
    return True


# ---------------------------------------------------------------------------
# CHECKOUT SESSION
# ---------------------------------------------------------------------------

def create_checkout_session(tier: str, customer_email: str, success_url: str, cancel_url: str) -> Optional[str]:
    """
    Create a Stripe Checkout Session and return the redirect URL.

    Args:
        tier:           One of 'professional', 'enterprise', 'enterprise_plus'
        customer_email: Pre-fill the checkout form
        success_url:    Where Stripe sends the user after payment
        cancel_url:     Where Stripe sends the user if they cancel

    Returns:
        Checkout URL string, or None on failure
    """
    if not _init_stripe():
        return None

    price_id = _get_price_id(tier)
    if not price_id:
        st.error(f"No Stripe price ID configured for tier '{tier}'. Add it to secrets.toml.")
        return None

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            customer_email=customer_email,
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={"tier": tier},
            billing_address_collection="required",
            allow_promotion_codes=True,
        )
        return session.url
    except Exception as e:
        st.error(f"Stripe error: {e}")
        return None


# ---------------------------------------------------------------------------
# CUSTOMER PORTAL (manage / cancel subscription)
# ---------------------------------------------------------------------------

def create_portal_session(customer_id: str, return_url: str) -> Optional[str]:
    """Return URL for Stripe Customer Portal (billing management)."""
    if not _init_stripe():
        return None
    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )
        return session.url
    except Exception as e:
        st.error(f"Stripe portal error: {e}")
        return None


# ---------------------------------------------------------------------------
# WEBHOOK HANDLER  (call from a Flask/FastAPI endpoint, not from Streamlit)
# ---------------------------------------------------------------------------

def handle_webhook(payload: bytes, sig_header: str) -> Dict:
    """
    Verify and process a Stripe webhook event.

    Usage (FastAPI example):
        @app.post("/stripe/webhook")
        async def stripe_webhook(request: Request):
            payload    = await request.body()
            sig_header = request.headers.get("stripe-signature")
            return handle_webhook(payload, sig_header)
    """
    keys = _get_stripe_keys()
    webhook_secret = keys.get("webhook_secret", "")

    if not _init_stripe() or not webhook_secret:
        return {"status": "error", "message": "Stripe not configured"}

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except stripe.error.SignatureVerificationError:
        return {"status": "error", "message": "Invalid signature"}

    event_type = event["type"]

    if event_type == "checkout.session.completed":
        session = event["data"]["object"]
        _on_checkout_completed(session)

    elif event_type == "customer.subscription.deleted":
        subscription = event["data"]["object"]
        _on_subscription_cancelled(subscription)

    elif event_type in ("invoice.payment_failed", "invoice.payment_action_required"):
        invoice = event["data"]["object"]
        _on_payment_failed(invoice)

    return {"status": "ok", "type": event_type}


def _on_checkout_completed(session):
    """Upgrade user tier after successful payment."""
    tier  = session.get("metadata", {}).get("tier")
    email = session.get("customer_email", "")
    customer_id = session.get("customer", "")

    if not tier or not email:
        return

    users_file = Path("data/users.json")
    if not users_file.exists():
        return

    with open(users_file) as f:
        users = json.load(f)

    # Find user by email and upgrade tier
    for username, data in users.items():
        if data.get("email", "").lower() == email.lower():
            data["tier"]        = tier
            data["stripe_customer_id"] = customer_id
            data["upgraded_at"] = datetime.now().isoformat()
            break

    with open(users_file, "w") as f:
        json.dump(users, f, indent=2)


def _on_subscription_cancelled(subscription):
    """Downgrade user to Free when subscription is cancelled."""
    customer_id = subscription.get("customer", "")

    users_file = Path("data/users.json")
    if not users_file.exists():
        return

    with open(users_file) as f:
        users = json.load(f)

    for username, data in users.items():
        if data.get("stripe_customer_id") == customer_id:
            data["tier"]         = "free"
            data["cancelled_at"] = datetime.now().isoformat()
            break

    with open(users_file, "w") as f:
        json.dump(users, f, indent=2)


def _on_payment_failed(invoice):
    """Log payment failures (send email via SendGrid in production)."""
    customer_id = invoice.get("customer", "")
    print(f"[Payment failed] customer={customer_id} – follow up needed")


# ---------------------------------------------------------------------------
# STREAMLIT UI COMPONENTS
# ---------------------------------------------------------------------------

def render_upgrade_button(tier: str, current_user_email: str, app_url: str):
    """
    Render a branded 'Upgrade' button for a specific tier.
    On click, redirects the user to Stripe Checkout.
    """
    tier_info = TIER_PRICES.get(tier, {})
    label     = f"Upgrade to {tier_info.get('name', tier)} — {tier_info.get('display', '')}"

    if not STRIPE_AVAILABLE or not _get_stripe_keys().get("secret_key"):
        # Graceful fallback: mailto link
        subject = f"{tier_info.get('name', tier)} Tier Inquiry – Encinitas"
        href    = f"mailto:sales@encinitas.ai?subject={subject}"
        st.markdown(
            f'<a href="{href}" target="_blank" style="'
            f'background:linear-gradient(90deg,#12dbb4,#14d8e2);'
            f'color:white;padding:12px 24px;border-radius:10px;'
            f'font-weight:600;text-decoration:none;display:inline-block;'
            f'font-size:0.95rem;">{label}</a>',
            unsafe_allow_html=True,
        )
        return

    if st.button(label, key=f"upgrade_{tier}", use_container_width=True):
        checkout_url = create_checkout_session(
            tier=tier,
            customer_email=current_user_email,
            success_url=f"{app_url}?payment=success&tier={tier}",
            cancel_url=f"{app_url}?payment=cancelled",
        )
        if checkout_url:
            st.markdown(
                f'<meta http-equiv="refresh" content="0; url={checkout_url}">',
                unsafe_allow_html=True,
            )


def render_payment_success_banner():
    """Show a success banner after Stripe redirects back."""
    params = st.query_params
    if params.get("payment") == "success":
        tier = params.get("tier", "")
        tier_name = TIER_PRICES.get(tier, {}).get("name", tier.title())
        st.success(
            f"🎉 Payment successful! Your account has been upgraded to **{tier_name}**. "
            "Refresh the page to access new features."
        )
        # Clear the query param so it doesn't show on every reload
        if st.button("Dismiss"):
            st.query_params.clear()
            st.rerun()
    elif params.get("payment") == "cancelled":
        st.info("Payment was cancelled. You can upgrade any time from the Pricing page.")
        if st.button("Dismiss"):
            st.query_params.clear()
            st.rerun()


def render_billing_portal_link(customer_id: str, app_url: str):
    """Render a 'Manage Billing' link for existing paid customers."""
    portal_url = create_portal_session(customer_id, return_url=app_url)
    if portal_url:
        st.markdown(
            f'<a href="{portal_url}" target="_blank" style="'
            f'color:#12dbb4;font-weight:600;text-decoration:none;">⚙️ Manage Billing</a>',
            unsafe_allow_html=True,
        )
