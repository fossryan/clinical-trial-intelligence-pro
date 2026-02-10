"""
Legal Disclaimers for Encinitas
Renders persistent banners, footer, and full Terms / Privacy pages
"""

import streamlit as st


# ---------------------------------------------------------------------------
# PERSISTENT DISCLAIMER BANNER  (rendered on every page)
# ---------------------------------------------------------------------------

def render_disclaimer_banner():
    """
    Render a compact, dismissible disclaimer banner at the top of every page.
    Once dismissed it stays hidden for the browser session.
    """
    if st.session_state.get("disclaimer_dismissed"):
        return

    st.markdown("""
    <style>
        .enc-disclaimer-banner {
            background: #fffbeb;
            border-left: 4px solid #f59e0b;
            border-radius: 0 8px 8px 0;
            padding: 10px 16px;
            font-size: 0.82rem;
            color: #78350f;
            margin-bottom: 12px;
            line-height: 1.5;
        }
        .enc-disclaimer-banner strong { color: #92400e; }
    </style>
    <div class="enc-disclaimer-banner">
        <strong>⚠️ Research &amp; Business Intelligence Only.</strong>
        Encinitas predictions are derived from statistical models trained on historical trial data
        and are intended solely for research, business intelligence, and portfolio-planning purposes.
        They do <strong>not</strong> constitute medical advice, clinical guidance, or investment advice,
        and must <strong>not</strong> be used for patient-care decisions or regulatory submissions.
        Accuracy is not guaranteed. Always consult qualified professionals before making clinical or
        financial decisions.
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([9, 1])
    with col_b:
        if st.button("✕ Dismiss", key="dismiss_disclaimer"):
            st.session_state["disclaimer_dismissed"] = True
            st.rerun()


# ---------------------------------------------------------------------------
# PERSISTENT FOOTER  (rendered on every page)
# ---------------------------------------------------------------------------

def render_footer():
    """Sticky footer with copyright, legal links and data-source note."""
    st.markdown("""
    <style>
        .enc-footer-bar {
            margin-top: 60px;
            padding: 18px 32px;
            background: #f8fafc;
            border-top: 1px solid #e2e8f0;
            font-size: 0.78rem;
            color: #94a3b8;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
            font-family: 'Inter', sans-serif;
        }
        .enc-footer-bar a {
            color: #12dbb4;
            text-decoration: none;
            margin-left: 14px;
        }
        .enc-footer-bar a:hover { text-decoration: underline; }
    </style>
    <div class="enc-footer-bar">
        <span>
            © 2026 Encinitas Clinical Intelligence, Inc. &nbsp;|&nbsp;
            Data source: ClinicalTrials.gov (NIH/NLM) — public domain &nbsp;|&nbsp;
            <strong>Not for clinical or patient-care use.</strong>
        </span>
        <span>
            <a href="?legal_page=terms"   target="_self">Terms of Service</a>
            <a href="?legal_page=privacy" target="_self">Privacy Policy</a>
            <a href="mailto:legal@encinitas.ai">Contact Legal</a>
        </span>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# FULL LEGAL PAGES  (Terms of Service & Privacy Policy)
# ---------------------------------------------------------------------------

def render_terms_of_service():
    """Full Terms of Service page."""
    st.title("📄 Terms of Service")
    st.caption("Last updated: February 9, 2026")
    st.markdown("---")

    st.markdown("""
## 1. Acceptance of Terms

By accessing or using the Encinitas Clinical Trial Intelligence platform ("Service"), you agree to
be bound by these Terms of Service ("Terms"). If you do not agree, do not use the Service.

---

## 2. Description of Service

Encinitas provides a software-as-a-service platform that applies machine-learning models to
publicly available clinical-trial data to produce **research and business-intelligence outputs**,
including trial-outcome predictions, competitive analytics, and portfolio-risk summaries.

---

## 3. NOT Medical, Clinical, or Investment Advice

> **IMPORTANT DISCLAIMER**

The outputs of the Service — including, without limitation, trial-success predictions, risk
scores, protocol recommendations, and financial estimates — are provided **for research and
business-intelligence purposes only**.

- They do **not** constitute medical advice, clinical guidance, pharmaceutical advice, or
  investment advice.
- They must **not** be used as the basis for patient-care decisions, prescribing decisions,
  regulatory submissions, or securities transactions.
- Encinitas is **not** a medical device, a licensed healthcare provider, a registered
  investment adviser, or a pharmaceutical consultancy.
- Predictions are probabilistic and inherently uncertain. Historical patterns do not guarantee
  future trial outcomes.

Users are solely responsible for verifying any information obtained through the Service and for
obtaining appropriate professional advice before making clinical, regulatory, or financial decisions.

---

## 4. Regulatory Classification

The Service is classified as a **business-intelligence / decision-support tool** and is **not**
intended to be classified as a medical device under the FDA, CE Mark, or equivalent regulatory
regimes. If your intended use requires regulatory approval in your jurisdiction, you are
responsible for obtaining such approval before use.

---

## 5. Data Sources and Accuracy

- The Service relies primarily on data sourced from ClinicalTrials.gov (National Library of
  Medicine, NIH), which is in the public domain.
- Encinitas does not guarantee the completeness, accuracy, or timeliness of underlying data.
- Model performance metrics (accuracy, AUC, etc.) are measured on historical test sets and may
  differ from real-world performance on unseen data.

---

## 6. Acceptable Use

You agree **not** to:

- Use the Service to make patient-care decisions or advise patients.
- Use the Service to generate regulatory submissions without independent expert review.
- Reverse-engineer, decompile, or extract model weights or training data.
- Share login credentials or allow unauthorised access.
- Use automated scraping or API abuse that disrupts the Service.
- Represent Encinitas outputs as independent professional opinions without proper attribution.

---

## 7. Intellectual Property

All model architectures, software, documentation, and UI designs are the exclusive intellectual
property of Encinitas Clinical Intelligence, Inc. The underlying clinical-trial data from
ClinicalTrials.gov remains in the public domain.

---

## 8. Limitation of Liability

TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, ENCINITAS SHALL NOT BE LIABLE FOR ANY
INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES ARISING FROM USE OF THE
SERVICE, INCLUDING BUT NOT LIMITED TO CLINICAL OUTCOMES, FINANCIAL LOSSES, OR REGULATORY
PENALTIES, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

---

## 9. Indemnification

You agree to indemnify and hold harmless Encinitas, its officers, directors, employees, and
agents from any claims, losses, or damages arising from your violation of these Terms or your
use of the Service.

---

## 10. Subscription, Payment, and Cancellation

- Paid subscriptions are billed annually in advance.
- Payments are processed by Stripe, Inc. Encinitas does not store card numbers.
- No refunds are issued for partial subscription periods unless required by applicable law.
- Encinitas reserves the right to modify pricing with 60 days' notice.
- Failure to pay may result in downgrade to the Free tier.

---

## 11. Confidentiality

Any proprietary trial data you upload remains yours. Encinitas will not share it with third
parties. Aggregated, anonymised usage statistics may be used to improve the Service.

---

## 12. Termination

Encinitas may suspend or terminate your access for violation of these Terms, non-payment, or
at its sole discretion with 30 days' notice. Upon termination, your data will be deleted within
90 days.

---

## 13. Governing Law

These Terms are governed by the laws of the State of Delaware, USA, without regard to conflict
of law principles.

---

## 14. Changes to Terms

Encinitas may update these Terms. Material changes will be notified by email or in-app banner
at least 30 days before taking effect.

---

## 15. Contact

Questions about these Terms: **legal@encinitas.ai**
    """)


def render_privacy_policy():
    """Full Privacy Policy page."""
    st.title("🔒 Privacy Policy")
    st.caption("Last updated: February 9, 2026")
    st.markdown("---")

    st.markdown("""
## 1. Introduction

Encinitas Clinical Intelligence, Inc. ("Encinitas", "we", "us") respects your privacy. This
Policy explains what data we collect, how we use it, and your rights.

---

## 2. Data We Collect

| Category | Examples | Purpose |
|---|---|---|
| Account data | Username, email, hashed password | Authentication |
| Usage data | Pages visited, predictions made, timestamps | Product improvement |
| Uploaded trial data | CSV files you upload | Your predictions only |
| Payment data | Billing name/address (via Stripe) | Subscription billing |
| Technical data | IP address, browser, OS | Security & analytics |

We do **not** collect:
- Plain-text passwords (passwords are SHA-256 hashed)
- Credit card numbers (handled exclusively by Stripe)
- Any patient health information (PHI)

---

## 3. HIPAA

The Service is designed to operate with **de-identified, publicly available** clinical-trial data.
We are **not** a HIPAA-covered entity for this Service. Do not upload PHI.

---

## 4. GDPR / CCPA

- **GDPR**: If you are in the EEA, you have rights of access, rectification, erasure, and
  portability. Contact **privacy@encinitas.ai** to exercise these rights.
- **CCPA**: California residents may request disclosure or deletion of personal data.

---

## 5. Data Sharing

We do **not** sell your personal data. We share data only with:

- **Stripe, Inc.** – for payment processing (their privacy policy applies).
- **GitHub** (if you use the CI/CD pipeline) – for automated workflows.
- **Law enforcement** – only when legally required.

---

## 6. Data Retention

- Account data: retained while your account is active, deleted within 90 days of cancellation.
- Uploaded trial data: deleted when you delete the file or close your account.
- Usage logs: retained for 12 months.

---

## 7. Security

- Passwords are stored as SHA-256 hashes (not plain text).
- Data in transit is encrypted with TLS 1.2+.
- Access controls restrict data to authorised personnel only.

---

## 8. Cookies

We use only essential session cookies required for authentication. No advertising or tracking
cookies.

---

## 9. Children

The Service is not directed at individuals under 18. We do not knowingly collect data from minors.

---

## 10. Contact

Privacy questions: **privacy@encinitas.ai**
    """)


# ---------------------------------------------------------------------------
# ROUTER  (call from main app to handle ?legal_page= query params)
# ---------------------------------------------------------------------------

def handle_legal_page_routing() -> bool:
    """
    Check if a legal page is requested via query param.
    Returns True if a legal page was rendered (so the main app can skip).

    Usage in streamlit_app.py:
        from legal import handle_legal_page_routing
        if handle_legal_page_routing():
            st.stop()
    """
    page = st.query_params.get("legal_page", "")

    if page == "terms":
        render_terms_of_service()
        st.markdown("---")
        if st.button("← Back to App"):
            st.query_params.clear()
            st.rerun()
        return True

    if page == "privacy":
        render_privacy_policy()
        st.markdown("---")
        if st.button("← Back to App"):
            st.query_params.clear()
            st.rerun()
        return True

    return False
