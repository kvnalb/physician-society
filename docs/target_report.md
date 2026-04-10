# Target report: Tirzepatide adoption simulation (research demo)

This document is the **intended narrative** for the asynchronous research demo: what problem we pose, **who** we model, **why** that sample is scoped the way it is, and **what** LLM-based physician simulation is—and is not—good for.

---

## 1. Backdrop

### The commercial situation (stylized)

Imagine **June 2022**: tirzepatide (Mounjaro) has recently reached the U.S. market as a new GLP-1–class option for type 2 diabetes. A competitor’s oral/injectable franchise (e.g., semaglutide) already dominates much of GLP-1 prescribing. **Brand and sales leadership** must allocate finite resources in a short window: which **physician segments** to defend, where to invest in **field force** and **KOL engagement**, and how to **tailor messaging** by prescriber archetype.

Traditional market research—national physician surveys, deep qual, longitudinal panels—can be **slow**, **expensive**, and **weakly anchored** to what physicians *actually do* in prescribing data. Hypothetical questions (“What would you do if a new agent launched?”) often **diverge** from later **revealed behavior**.

### The analytic opportunity

Administrative **Medicare Part D** data offer a **repeatable, scalable** view of **prescribing mix** at the physician level, before and after a launch window. When linked to **registry** attributes (specialty, geography) and **Open Payments** exposure, they support **grounded personas**: not imaginary doctors, but **specific NPIs** summarized by **observed** professional context and **observed** prescribing.

**LLM-based simulation** then lets us **elicit structured judgments** (e.g., single-select survey items about diabetes prescribing and adoption posture) **at scale**, compare **alternative persona-construction methods**, and optionally contrast elicited responses with **behavioral outcomes** measured in the **next** data year.

This demo is a **proof of concept** for that workflow—not a production forecast for a single brand.

---

## 2. The sample: scope conditions, not apologies

### What we study

We work with a **fixed cohort** of roughly **100 physicians** drawn through a **transparent, purposive pipeline**:

- **Geography:** Practice locations in a defined set of **large U.S. metros** in selected states (commercially salient geographies for many launch plans).
- **Clinical role:** Individual prescribers (**NPPES** type-1) in **diabetes-focused specialties** (endocrinology, internal medicine, family medicine; **cardiology excluded** in the cohort script to sharpen GLP‑1/diabetes narrative and balance strata).
- **Prescribing relevance:** **Medicare Part D 2022** aggregates show **meaningful diabetes-related prescribing** (volume and share thresholds) so the unit is not a marginal prescriber in the category.
- **Behavioral readout:** The same NPIs appear in **Part D 2023**, so we can attach **post-window** measures (e.g., GLP-1–class activity, tirzepatide-related claims where identifiable) as **scope-appropriate “ground truth”** for **revealed prescribing**, not for attitudes.

### Why this is not a “representative U.S. physician” sample

We **do not** claim that this cohort is a **probability sample** of all U.S. physicians. National representativeness is the wrong target for many **launch-tactical** questions, which are inherently **segment- and geography-focused**.

### Why it can still be **decision-relevant**

For **category-level launch decisions**, stakeholders often care most about **prescribers who actually drive relevant volume** in **priority markets**—not about the modal U.S. physician who rarely touches the class. Our frame is **aligned with that decision unit**:

- **Dense data** in the lens we use (Part D–visible prescribing).
- **Explicit segment heterogeneity** (specialty, metro, derived archetypes, pharma engagement tiers).
- **Linked pre/post years** for **behavioral** comparison, within the same scoped population.

**Interpretation rule:** Conclusions apply **first** to **this defined slice**. Broader implications are **hypothesis-generating** and should be **validated** with scaled claims analytics, commercial data, or primary research—not assumed to transfer without evidence.

### Medicare Part D as a scope condition

We frame **Medicare Part D** not as an “unfortunate limitation” but as the **observed channel** for prescribing in this study:

- **In scope:** Behavior visible in **Part D** for physicians meeting inclusion rules.
- **Out of scope:** Non-Medicare channels, cash pay, and prescribing invisible in these extracts.

That boundary should be **stated once clearly** and then used consistently in all charts and copy.

---

## 3. What we simulate and how (high level)

### Personas

Each run uses **structured prompts** built from the same cohort row. We compare at least **two methods**:

- **Rich persona:** Many **observed** fields (prescribing mix, engagement tier, archetype labels, etc.).
- **Minimal persona:** A **sparse** subset (e.g., specialty, geography, archetype) to test how much **context** is needed for stable elicitation.

### Instrument

A **small battery** of **single-select** items—worded like **clinical–commercial judgment** (e.g., second-line preferences, GLP-1 penetration bands, adoption posture, branded vs generic leaning). Responses are constrained to **machine-parseable** option IDs so we can **aggregate distributions** and **compare methods**.

### Evaluation (pre-defined at concept level)

- **Method comparison:** Agreement between rich vs minimal elicitation (e.g., Cohen’s \(\kappa\) on paired NPI–question responses), plus per-question distribution contrasts.
- **Behavioral readout (descriptive):** **Empirical** adoption and class dynamics in **2023 Part D** for the same cohort—reported as **baselines** and segment breakdowns, not as proof of causal impact of any single factor.

We **avoid** claiming **national predictive validation** or **causal** effects of industry payments on prescribing unless the design explicitly supports it.

---

## 4. What this simulation is **useful** for

| Use case | Rationale |
|----------|-----------|
| **Fast segment hypotheses** | Explore how **different persona constructions** shift aggregate answers on the **same** questions—useful for **sensitivity analysis** of “what we think doctors would say” under alternative grounding. |
| **Narrative and messaging stress-tests** | Test **competing frames** (within guardrails) across **heterogeneous** synthetic jurors grounded in **real features**, before expensive human fieldwork. |
| **Reasoning traces (when captured)** | LLM outputs can include **short rationales** for audit and **qualitative coding**, not as evidence of real physician cognition but as **traceable model behavior**. |
| **Coupling judgment to claims** | When questions are **mapped carefully** to measurable constructs, we can **contrast** elicited patterns with **revealed** prescribing in the **scoped** data—highlighting **where stated and revealed signals diverge** (a productive research direction). |
| **Enterprise-shaped workflow** | The intended product path mirrors **artificial societies** at a smaller scale: **grounded personas → structured instruments → aggregate + segment readouts → iteration** under configuration control (models, prompts, temperature). |

---

## 5. What this simulation is **not** (boundaries)

- **Not** a substitute for **regulatory-grade** evidence or **MLR-reviewed** promotional claims.
- **Not** guaranteed **calibration** to **national** physician opinion distributions.
- **Not** **causal identification** of how payments, access, or detailing **change** prescribing without an explicit design.
- **Not** a **peer social network** model unless **edges** are supplied or estimated from data we do not claim to have here.

---

## 6. Why this matters for an “Artificial Societies”–style product

Enterprise clients often need to **test sensitive or fast-moving narratives** across **many synthetic stakeholders** grounded in **real-world segments**. This demo shrinks that idea to a **transparent, auditable** slice:

- **Real identifiers and real administrative features** (within public data rules).
- **Fixed instruments** and **documented methods** so results can be **re-run** and **compared**.
- **Scope conditions** stated up front so buyers know **exactly which population** the readout refers to.

The path from here to production is **scale** (more personas, more questions, richer data contracts), **governance** (PII, compliance, review workflows), and **validation** programs—not a different **conceptual** foundation.

---

## 7. One-sentence summary

> We **scope** the study to **Medicare Part D–visible, diabetes-active prescribers in priority metros**, elicit **structured judgments** with **LLM personas** built at **two richness levels**, and **compare** those elicitations to each other and to **observed 2023 prescribing**—to support **fast, segment-aware, hypothesis-generating** launch analytics without pretending we sampled **all U.S. physicians**.

---

*This file is the **target** narrative for the final demo write-up; implementation artifacts (Streamlit app, `artifacts/demo/`, README) should stay consistent with these scope conditions.*
