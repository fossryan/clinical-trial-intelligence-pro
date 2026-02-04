# Clinical Trial Risk Intelligence — SaaS Business Strategy

## Can This Be Sold? Short Answer: Yes. Here's Exactly How.

---

## 1. THE MARKET OPPORTUNITY

### Who Pays for This (and How Much)

Clinical trial failure is a $150B+ annual problem in pharma.  
90% of drugs fail in trials.  A single Phase 3 failure costs $500M–$2B.  
Companies spend $200M–$500M annually on risk analytics and portfolio tools.

The buyers are not scientists — they are **BD directors, portfolio managers, and CFOs** who need to answer: *"Which of our 40 trials is most likely to fail in the next 12 months?"*

Existing tools in this space include Insilico Medicine, Recursion, BenevolentAI, and Veeva Vault (clinical module). Most are $50K–$500K/year enterprise contracts. The gap this platform fills: a **fast, self-service, data-ingestible risk scorer** — not a $2M consulting engagement.

### Target Customer Segments (Ranked by Ease of Sale)

**Tier 1 — Biotech startups (fastest to close, 2–8 weeks):**  
5–50 employees, 2–10 trials in portfolio, limited in-house analytics.  
They can't afford a Veeva-scale platform.  They need answers in days, not months.  
Budget: $500–$5,000/month.

**Tier 2 — Mid-size pharma BD teams (medium cycle, 4–12 weeks):**  
Companies like Regeneron, BioMarin, Neurocrine.  BD teams evaluating licensing deals and need to score target company trial portfolios in hours.  
Budget: $1,000–$15,000/month.

**Tier 3 — VC / life-science investors (fast cycle for pilot, 2–6 weeks):**  
Due diligence analysts need to score 20 pipeline assets before a funding round.  Upload CSVs, get ranked risk reports.  Price sensitivity is low — this saves weeks of manual work.  
Budget: $200–$2,000/month.

**Tier 4 — Big Pharma R&D ops (long cycle, 3–6 months, but biggest contract):**  
Pfizer, Merck, BMS.  Enterprise integration, SOC 2, HIPAA-adjacent compliance.  These are the logos that make the company credible but are hardest to close first.  
Budget: $25,000–$150,000/year.

---

## 2. PRICING MODEL (Three Tiers)

### Starter — $299/month
- Upload up to **50 trials/month**
- Batch predictions + downloadable reports
- Public benchmark comparison
- Email support
- *Target: Biotech founders, solo consultants, VC analysts doing quick screens*

### Pro — $999/month
- Up to **500 trials/month**
- Portfolio Analyzer with custom dashboards
- Indication-level benchmarking
- Priority support (24h response)
- API access (10,000 predictions/month)
- CSV and JSON export
- *Target: Biotech BD teams, mid-size pharma, VC firms*

### Enterprise — $4,999/month (custom quotes for annual)
- **Unlimited** trial uploads
- Dedicated data integration (their CRM/ERP → your pipeline)
- Custom model retraining on client historical data
- SSO / SAML authentication
- SOC 2 Type II compliance
- Dedicated account manager
- White-label option (their logo, your engine)
- *Target: Big Pharma, CROs, large PE/VC funds*

### Pricing Rationale

The value to the customer is not the software — it is **one prevented trial failure**.  A single Phase 3 failure costs $500M+.  At $999/month the ROI is essentially infinite the moment it catches one bad trial.  This means pricing power is high and churn should be very low.

---

## 3. WHAT NEEDS TO BE BUILT TO SELL THIS

The current platform is a working proof-of-concept.  To sell it as a product, here is the honest gap list, in priority order:

### Must-Have (before first paying customer)

**1. Cloud hosting** — move off local Streamlit.  Deploy on AWS or GCP.  Streamlit Cloud handles this for free up to a point, then ~$50–250/month.  Alternatively, containerise with Docker and deploy on a VPS.

**2. User accounts and authentication** — right now anyone who can run the script can use it.  Need login/signup, email verification, password reset.  Firebase Auth or Auth0 handles this in a weekend.

**3. Data isolation** — each customer's uploaded CSVs must be stored and predicted in their own namespace.  No customer can see another's data.  This is both a legal requirement and a trust requirement.

**4. Rate limiting and usage tracking** — the pricing tiers are defined by trial upload volume.  Need a counter per account per billing period.

**5. Payment processing** — Stripe integration.  Monthly recurring billing.  Invoice generation.  This is a full weekend of work but well-documented.

**6. Terms of Service and Privacy Policy** — non-negotiable before accepting payment.  A lawyer review runs $2K–$5K.  Many startups use templates from services like Clerky or Stripe Atlas to cut this cost.

### Should-Have (before scaling past 10 customers)

**7. API endpoint** — the Pro tier promises API access.  This means wrapping the prediction engine in a REST API (FastAPI is the obvious choice) with API key auth and per-key rate limits.

**8. Audit logging** — who uploaded what, when predictions ran, what was downloaded.  Enterprise customers will ask for this.

**9. Email notifications** — "Your batch prediction is ready," "Your monthly usage is at 80%," etc.  SendGrid or AWS SES.

**10. Dashboard customisation** — let customers save filter states, create named portfolios, export charts as images.

### Nice-to-Have (enterprise tier)

**11. SSO/SAML** — big pharma IT departments will not approve a product without it.

**12. Custom model retraining** — if a client has 5 years of their own trial outcomes, retrain the model on their data for higher accuracy on their specific indication mix.  This is a major differentiator and justifies the Enterprise price.

**13. CRM/ERP connectors** — auto-ingest trials from Veeva, Salesforce, or internal systems via API.

---

## 4. LEGAL AND COMPLIANCE CONSIDERATIONS

### Data Ownership
The customer's uploaded trial data is **their property**.  Your Terms of Service must state this explicitly.  You do not own it, you do not train your public model on it without explicit consent, and you delete it if they cancel.

### HIPAA
Clinical trial *design metadata* (phase, enrollment, sponsor, indication) is **not** Protected Health Information (PHI).  Individual patient data would be PHI.  As long as your upload template contains only trial-level aggregates — which it does — you are **not** a Covered Entity under HIPAA and do not need a BAA (Business Associate Agreement).

If you later add patient-level data, this changes entirely.  Do not cross that line without a compliance lawyer.

### GDPR / Data Residency
European pharma customers will ask where their data is stored.  Offer an EU data residency option (AWS eu-west-1 or similar).  This is a hosting decision, not a code decision.

### Intellectual Property
The prediction model and the platform code are **your** IP.  The customer's uploaded data is **their** IP.  This boundary must be clear in the contract.

### Insurance
Once you have paying customers, get a **Cyber Liability** policy.  Cost: $1K–$5K/year for a small SaaS.  VCs and enterprise buyers will ask if you have it.

---

## 5. GO-TO-MARKET STRATEGY

### Phase 1 — First 3 Customers (Months 1–3)

**How to find them:**

Your network is the fastest path.  Biotech accelerators (Y Combinator, MATTER, BioAtla), LinkedIn (search "clinical ops manager" or "BD analytics"), and Reddit/Twitter pharma communities all surface potential early adopters.

**What to offer:**
A **free pilot** — 3 months, unlimited uploads, no credit card.  In exchange, you get:
- Permission to use them as a case study (anonymised)
- 30 minutes of feedback after month 2
- A reference call if you ask

**Goal:** 3 pilots → 1 conversion.  One paying customer validates the model.

### Phase 2 — Product-Market Fit (Months 3–9)

**Content marketing:**
Write 3–5 blog posts and LinkedIn articles.  Topics that actually get clicks in this space:

- "The 5 features that predict Phase 2 trial failure (data from 2,000 trials)"
- "How to score a biotech's pipeline in 10 minutes before a funding call"
- "Why 60% of oncology trials fail and what the data actually shows"

These are SEO-friendly, shareable, and position you as the credible data source.

**Partnership channel:**
Approach 2–3 small CROs (Contract Research Organisations) or biotech consulting firms.  Offer them a white-label or referral arrangement.  They have the customer relationships; you have the tool.

### Phase 3 — Scale (Months 9–18)

Once you have 20+ customers and product-market fit signals (low churn, NPS > 40), raise a small seed round ($500K–$2M) to hire:
- One sales person (BD background, not tech)
- One engineer (to build the API and enterprise features)

Target $100K ARR (Annual Recurring Revenue) by month 18.  This is achievable with 10 Pro customers or 2 Enterprise customers.

---

## 6. COMPETITIVE POSITIONING

### What makes this defensible?

**Speed:** Most competitors are 6–12 month implementation cycles.  This is upload-and-go in minutes.

**Transparency:** The model explains *why* a trial is high risk (the risk factor breakdown).  Black-box AI tools frustrate BD teams — they need to justify decisions to boards.

**Benchmark data:** The built-in 2,000-trial public benchmark is included free.  Competitors charge separately for benchmark databases.

**Price:** $299–$999/month vs $50K–$500K/year for enterprise competitors.  This is not a luxury tool — it's a utility.

### Where you are NOT competitive (be honest about this)

- Against Veeva Vault if the customer already has a full clinical trial management system
- Against custom ML teams at Big Pharma that have 100× your data
- On regulatory submission workflows (this is a risk scoring tool, not a regulatory tool)

---

## 7. THE PITCH DECK STRUCTURE (If You Ever Fundraise)

1. **Hook:** "90% of drugs fail in clinical trials. We predict which ones — before the company spends $500M finding out."
2. **Problem:** Phase 2/3 failure rate, cost per failure, speed of existing tools
3. **Solution:** 30-second demo of upload → prediction → portfolio view
4. **Data:** 78% accuracy, 2,000-trial benchmark, SHAP explainability
5. **Market:** $150B+ annual trial failure cost, 10,000+ biotechs as potential customers
6. **Business model:** 3-tier SaaS, $299–$4,999/month, land-and-expand
7. **Traction:** X pilots, Y paying customers, Z ARR (fill in when you have them)
8. **Team:** Your background (CAD/data engineering + biotech portfolio projects)
9. **Ask:** $500K seed to hire sales + engineer, target $100K ARR in 18 months

---

## 8. QUICK-START CHECKLIST

If you want to actually launch this as a product, here is the sequence:

- [ ] Deploy on Streamlit Cloud (free tier) — 30 minutes
- [ ] Set up Firebase Auth for user login — 2 hours
- [ ] Add per-user data storage (Firebase Firestore or S3) — 4 hours
- [ ] Integrate Stripe for billing — 4 hours
- [ ] Write Terms of Service (use a template, get a lawyer to review) — 1 week
- [ ] Create a landing page (even a single Carrd or Vercel page) — 2 hours
- [ ] Find 3 pilot customers — 2–4 weeks
- [ ] Ship to first pilot — Day 1 of pilots
- [ ] Collect feedback, iterate — Weeks 2–4
- [ ] Convert first paid customer — Week 5–8

**Total time to first paying customer: 6–10 weeks if you treat it as a side project alongside your day job.**

---

*This document is a strategy guide, not legal or financial advice.  Consult a lawyer before accepting payment for a SaaS product and before making any business decisions based on market projections.*
