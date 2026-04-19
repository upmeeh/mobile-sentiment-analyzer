import streamlit as st
import pickle
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(script_dir, 'vectorizer.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)

ASPECTS = {
    "Battery": ["battery", "charge", "charging", "drain", "life"],
    "Camera": ["camera", "photo", "picture", "lens", "zoom", "selfie"],
    "Display": ["screen", "display", "brightness", "amoled", "resolution"],
    "Performance": ["speed", "fast", "slow", "lag", "processor", "performance", "hang"],
    "Price": ["price", "worth", "value", "expensive", "cheap", "cost"],
    "Build": ["build", "design", "quality", "finish", "plastic", "glass", "feel"],
}

ASPECT_ICONS = {
    "Battery": "PWR",
    "Camera": "CAM",
    "Display": "DSP",
    "Performance": "CPU",
    "Price": "VAL",
    "Build": "BLD",
}

def get_aspect_sentiments(review):
    sentences = (
        review
        .replace("!", ".")
        .replace("?", ".")
        .replace(" but ", ". ")
        .replace(" however ", ". ")
        .replace(" although ", ". ")
        .split(".")
    )
    results = {}
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        for aspect, keywords in ASPECTS.items():
            if any(word in sentence.lower() for word in keywords):
                vec = vectorizer.transform([sentence])
                pred = model.predict(vec)[0]
                results[aspect] = pred
    return results

st.set_page_config(page_title="PhoneSense", page_icon="📱", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Ndot+57:wght@400&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
    background-color: #F2F0EB;
    font-family: 'Space Mono', monospace;
}

section[data-testid="stMain"] > div {
    padding-top: 0 !important;
    max-width: 720px;
    margin: 0 auto;
}

/* dot grid background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: radial-gradient(circle, #00000018 1px, transparent 1px);
    background-size: 24px 24px;
    pointer-events: none;
    z-index: 0;
}

.block-container { position: relative; z-index: 1; }

/* ---- NAV BAR ---- */
.nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.2rem 0;
    border-bottom: 1px solid #00000022;
    margin-bottom: 3rem;
}

.nav-logo {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.3em;
    color: #111;
    text-transform: uppercase;
}

.nav-tag {
    font-size: 10px;
    letter-spacing: 0.2em;
    color: #888;
}

/* ---- HERO ---- */
.hero-eyebrow {
    font-size: 10px;
    letter-spacing: 0.3em;
    color: #888;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: clamp(2.8rem, 8vw, 5rem);
    font-weight: 700;
    color: #111;
    line-height: 0.95;
    letter-spacing: -0.02em;
    margin-bottom: 1.5rem;
}

.hero-title span {
    display: inline-block;
    border-bottom: 3px solid #111;
}

.hero-desc {
    font-size: 11px;
    letter-spacing: 0.15em;
    color: #666;
    text-transform: uppercase;
    margin-bottom: 3rem;
}

/* ---- GRID RULE ---- */
.grid-rule {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    border-top: 1px solid #00000022;
    border-left: 1px solid #00000022;
    margin-bottom: 2rem;
}

.grid-cell {
    border-right: 1px solid #00000022;
    border-bottom: 1px solid #00000022;
    padding: 0.8rem 1rem;
    font-size: 9px;
    letter-spacing: 0.2em;
    color: #999;
    text-transform: uppercase;
}

/* ---- INPUT SECTION ---- */
.input-label {
    font-size: 9px;
    letter-spacing: 0.25em;
    color: #999;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

div[data-testid="stTextArea"] label { display: none !important; }

textarea {
    background: #FFFFFF !important;
    border: 1px solid #00000033 !important;
    border-radius: 0 !important;
    color: #111 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
    letter-spacing: 0.05em !important;
    resize: none !important;
}

textarea:focus {
    border-color: #111 !important;
    box-shadow: none !important;
    outline: none !important;
}

textarea::placeholder { color: #bbb !important; }

/* ---- BUTTON ---- */
div[data-testid="stButton"] button {
    background: #111 !important;
    color: #F2F0EB !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    letter-spacing: 0.25em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.9rem 2rem !important;
    width: 100% !important;
    margin-top: 2px !important;
    transition: background 0.15s !important;
}

div[data-testid="stButton"] button:hover {
    background: #333 !important;
}

/* ---- RESULTS ---- */
.result-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    border-top: 1px solid #00000022;
    padding-top: 2rem;
    margin-top: 2rem;
    margin-bottom: 1.5rem;
}

.result-header-label {
    font-size: 9px;
    letter-spacing: 0.3em;
    color: #999;
    text-transform: uppercase;
    flex: 1;
}

.result-header-line {
    flex: 3;
    height: 1px;
    background: #00000015;
}

.verdict-block {
    padding: 2rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    border: 1px solid #00000022;
}

.verdict-pos { background: #111; }
.verdict-neg { background: #F2F0EB; border: 1px solid #111; }

.verdict-word {
    font-family: 'Space Mono', monospace;
    font-size: clamp(2rem, 6vw, 3.5rem);
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1;
}
.verdict-pos .verdict-word { color: #F2F0EB; }
.verdict-neg .verdict-word { color: #111; }

.verdict-meta {
    font-size: 9px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}
.verdict-pos .verdict-meta { color: #888; }
.verdict-neg .verdict-meta { color: #666; }

/* ---- ASPECT GRID ---- */
.aspect-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    border-top: 1px solid #00000022;
    border-left: 1px solid #00000022;
}

.aspect-cell {
    border-right: 1px solid #00000022;
    border-bottom: 1px solid #00000022;
    padding: 1.2rem 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

.aspect-code {
    font-size: 9px;
    letter-spacing: 0.25em;
    color: #999;
}

.aspect-name-text {
    font-size: 15px;
    font-weight: 700;
    color: #111;
    letter-spacing: -0.01em;
}

.aspect-result {
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
    padding: 3px 8px;
    display: inline-block;
    width: fit-content;
}

.apos { background: #111; color: #F2F0EB; }
.aneg { background: transparent; color: #111; border: 1px solid #111; }

/* ---- FOOTER ---- */
.footer {
    margin-top: 4rem;
    padding: 1.5rem 0;
    border-top: 1px solid #00000022;
    display: flex;
    justify-content: space-between;
    font-size: 9px;
    letter-spacing: 0.2em;
    color: #bbb;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# Nav
st.markdown("""
<div class="nav">
    <div class="nav-logo">Phone(Sense)</div>
    <div class="nav-tag">SYS_READY ●</div>
</div>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero-eyebrow">// sentiment analysis</div>
<div class="hero-title">READ THE<br><span>REVIEW.</span></div>
<div class="hero-desc">Mobile phone review analyzer — battery · camera · display · build</div>
""", unsafe_allow_html=True)

# Grid info strip
st.markdown("""
<div class="grid-rule">
    <div class="grid-cell">Model — LogReg</div>
    <div class="grid-cell">Vectorizer — TF-IDF</div>
    <div class="grid-cell">Dataset — 67K reviews</div>
</div>
""", unsafe_allow_html=True)

# Input
st.markdown('<div class="input-label">// input review</div>', unsafe_allow_html=True)
user_input = st.text_area("", placeholder="camera is great but battery drains too fast...", height=120, label_visibility="collapsed")
analyze = st.button("ANALYZE →")

# Results
if analyze:
    if user_input.strip() == "":
        st.warning("Input a review first.")
    else:
        vec = vectorizer.transform([user_input])
        overall = model.predict(vec)[0]

        is_pos = overall == "positive"
        verdict_css = "verdict-pos" if is_pos else "verdict-neg"
        verdict_word = "POSITIVE" if is_pos else "NEGATIVE"
        verdict_meta = "SIGNAL_POS // REVIEW_FAVORABLE" if is_pos else "SIGNAL_NEG // REVIEW_CRITICAL"

        st.markdown(f"""
        <div class="result-header">
            <div class="result-header-label">// verdict</div>
            <div class="result-header-line"></div>
        </div>
        <div class="verdict-block {verdict_css}">
            <div class="verdict-word">{verdict_word}</div>
            <div class="verdict-meta">{verdict_meta}</div>
        </div>
        """, unsafe_allow_html=True)

        aspects = get_aspect_sentiments(user_input)
        if aspects:
            st.markdown("""
            <div class="result-header">
                <div class="result-header-label">// aspect breakdown</div>
                <div class="result-header-line"></div>
            </div>
            """, unsafe_allow_html=True)

            cells = ""
            for aspect, sentiment in aspects.items():
                code = ASPECT_ICONS.get(aspect, "???")
                badge_class = "apos" if sentiment == "positive" else "aneg"
                badge_text = "POSITIVE" if sentiment == "positive" else "NEGATIVE"
                cells += f"""
                <div class="aspect-cell">
                    <div class="aspect-code">{code}_MODULE</div>
                    <div class="aspect-name-text">{aspect}</div>
                    <div class="aspect-result {badge_class}">{badge_text}</div>
                </div>
                """

            st.markdown(f'<div class="aspect-grid">{cells}</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <span>PhoneSense © 2025</span>
    <span>Built with TF-IDF + LogReg</span>
</div>
""", unsafe_allow_html=True)