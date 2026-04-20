from flask import Flask, request, render_template_string
import pickle
import os

app = Flask(__name__)

# ---- Load model ----
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(script_dir, 'vectorizer.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)

# ---- Aspect logic (same as yours) ----
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
        review.replace("!", ".")
        .replace("?", ".")
        .replace(" but ", ". ")
        .replace(" however ", ". ")
        .replace(" although ", ". ")
        .split(".")
    )
    results = {}
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        for aspect, keywords in ASPECTS.items():
            if any(w in s.lower() for w in keywords):
                vec = vectorizer.transform([s])
                pred = model.predict(vec)[0]
                results[aspect] = pred
    return results

# ---- HTML (same) ----
HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Smartphone Review Classification</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    background: #F2F0EB;
    font-family: 'Space Mono', monospace;
    color: #111;
}

/* subtle grid */
body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: radial-gradient(circle, #00000018 1px, transparent 1px);
    background-size: 24px 24px;
    pointer-events: none;
}

.container {
    max-width: 720px;
    margin: 0 auto;
    padding: 20px;
}

/* nav */
.nav {
    display: flex;
    justify-content: space-between;
    border-bottom: 1px solid #00000022;
    padding-bottom: 10px;
    margin-bottom: 30px;
}

.nav-logo { font-weight: 700; letter-spacing: 0.3em; font-size: 13px; }
.nav-tag { font-size: 10px; color: #888; }

/* hero */
.hero-eyebrow { font-size: 10px; letter-spacing: 0.3em; color: #888; margin-bottom: 10px; }
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 15px;
}
.hero-title span { border-bottom: 3px solid #111; }
.hero-desc { font-size: 11px; letter-spacing: 0.15em; color: #666; margin-bottom: 30px; }

/* grid rule */
.grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    border: 1px solid #00000022;
    margin-bottom: 20px;
}
.grid div {
    border-right: 1px solid #00000022;
    padding: 10px;
    font-size: 10px;
}
.grid div:last-child { border-right: none; }

/* input */
textarea {
    width: 100%;
    height: 120px;
    border: 1px solid #00000033;
    padding: 10px;
    font-family: 'Space Mono';
}
button {
    width: 100%;
    padding: 12px;
    margin-top: 10px;
    background: #111;
    color: white;
    border: none;
    cursor: pointer;
}

/* verdict */
.verdict {
    margin-top: 30px;
    padding: 20px;
    border: 1px solid #00000022;
    display: flex;
    justify-content: space-between;
}
.pos { background: #111; color: #F2F0EB; }
.neg { background: #F2F0EB; border: 1px solid #111; }
.neu { background: #F2F0EB; border: 1px solid #aaa; color: #888; }

/* aspects */
.aspect-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    border: 1px solid #00000022;
    margin-top: 20px;
}
.aspect {
    padding: 15px;
    border-right: 1px solid #00000022;
    border-bottom: 1px solid #00000022;
}
.aspect:nth-child(2n) { border-right: none; }

.badge {
    margin-top: 5px;
    display: inline-block;
    padding: 3px 8px;
    font-size: 10px;
}
.apos { background:#111; color:white; }
.aneg { border:1px solid #111; }
.aneu { border:1px solid #aaa; color:#888; }

.footer {
    margin-top: 40px;
    border-top: 1px solid #00000022;
    padding-top: 10px;
    font-size: 10px;
    display: flex;
    justify-content: space-between;
}
</style>
</head>

<body>
<div class="container">

<div class="nav">
    <div class="nav-logo">Smartphone Review Classification</div>
    <div class="nav-tag">SYS_READY ●</div>
</div>

<div class="hero-eyebrow">// sentiment analysis</div>
<div class="hero-title">READ THE<br><span>REVIEW.</span></div>
<div class="hero-desc">Mobile review analyzer</div>

<div class="grid">
    <div>Model — LogReg</div>
    <div>Vectorizer</div>
    <div>Dataset — Reviews</div>
</div>

<form method="POST">
    <textarea name="review" placeholder="camera is great but battery drains fast..."></textarea>
    <button>ANALYZE →</button>
</form>

{% if result %}
<div class="verdict {{css}}">
    <div>{{word}}</div>
    <div>{{confidence}}%</div>
</div>
{% endif %}

{% if aspects %}
<div class="aspect-grid">
{% for a, val in aspects.items() %}
<div class="aspect">
    <div>{{a}}</div>
    <div class="badge {{val_class[val]}}">{{val}}</div>
</div>
{% endfor %}
</div>
{% endif %}

<div class="footer">
    <span>Tejash</span>
    <span>Flask Version</span>
</div>

</div>
</body>
</html>)
"""

@app.route("/", methods=["GET","POST"])
def home():
    result = None
    confidence = None
    aspects = None
    css = ""
    word = ""

    val_class = {"positive":"apos","negative":"aneg","neutral":"aneu"}

    if request.method == "POST":
        text = request.form["review"]

        vec = vectorizer.transform([text])

        # ✅ NEW: probability FIRST
        proba = model.predict_proba(vec)[0]
        confidence = round(max(proba) * 100)

        # ✅ NEW: neutral logic
        if max(proba) < 0.6:
            result = "neutral"
        else:
            result = model.predict(vec)[0]

        # UI mapping
        if result == "positive":
            css = "pos"
            word = "POSITIVE"
        elif result == "negative":
            css = "neg"
            word = "NEGATIVE"
        else:
            css = "neu"
            word = "NEUTRAL"

        # ✅ FIXED indentation
        aspects = get_aspect_sentiments(text)

    return render_template_string(
        HTML,
        result=result,
        confidence=confidence,
        aspects=aspects,
        css=css,
        word=word,
        val_class=val_class
    )

app.run(host="0.0.0.0", port=3000)