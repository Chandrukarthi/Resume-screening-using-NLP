import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re

from utils import extract_text_from_pdf
from skills import extract_skills

# ── CONFIG ────
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("Resume Screening ")

# ── LOAD MODEL ─────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ── JOB DESCRIPTION INPUT ──────
st.subheader("📝 Job Description")

jd_option = st.radio("Choose input method:", ["Upload File", "Type Manually"])

job_desc = ""

def read_file(file):
    if file.name.endswith(".pdf"):
        with open("temp.pdf", "wb") as f:
            f.write(file.getbuffer())
        return extract_text_from_pdf("temp.pdf")
    return file.read().decode("utf-8", errors="ignore")

if jd_option == "Upload File":
    jd_file = st.file_uploader("Upload Job Description (TXT/PDF)", type=["txt", "pdf"])
    if jd_file:
        job_desc = read_file(jd_file)
else:
    job_desc = st.text_area("Enter Job Description here:", height=200)

# ── EXTRACTION FUNCTIONS ───────────────────────────
def extract_email(text):
    emails = re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return emails[0] if emails else "Not Found"

def extract_name(text):
    lines = text.strip().split("\n")
    for line in lines[:5]:
        line = line.strip()
        if len(line.split()) <= 4 and line.replace(" ", "").isalpha():
            return line
    return "Unknown"

# ── FILE UPLOAD ────────────────────────────────────
st.subheader("📂 Upload Resumes")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT resumes",
    accept_multiple_files=True
)

# ── ANALYZE BUTTON ─────────────────────────────────
analyze = st.button("▶ Analyze Resumes")

# ── PROCESS ────────────────────────────────────────
if analyze and uploaded_files and job_desc.strip() != "":

    with st.spinner("Analyzing resumes... ⏳"):

        texts = []
        names = []

        for file in uploaded_files:
            try:
                text = read_file(file)
                texts.append(text)
                names.append(file.name)
            except:
                st.warning(f"Error reading {file.name}")

        # ── EMBEDDINGS ─────────────────────────────
        jd_emb = model.encode([job_desc])
        res_emb = model.encode(texts)

        scores = cosine_similarity(jd_emb, res_emb)[0]

        jd_skills = extract_skills(job_desc)

        results = []

        for i in range(len(names)):
            score = scores[i] * 100

            email = extract_email(texts[i])
            person_name = extract_name(texts[i])

            resume_skills = extract_skills(texts[i])
            matched_skills = list(set(resume_skills) & set(jd_skills))

            results.append({
                "file_name": names[i],
                "candidate_name": person_name,
                "email": email,
                "score": round(score, 2),
                "matched_skills": matched_skills
            })

        # Sort results
        results = sorted(results, key=lambda x: x["score"], reverse=True)

    # ── RESULTS ─────────────────────────────────
    st.success("✅ Analysis Complete")

    col1, col2 = st.columns(2)
    col1.metric("Candidates", len(results))

    if results:
        col2.metric("Top Match", f"{results[0]['score']:.2f}%")

    st.divider()

    # ── RANKING ────────────────────────────────
    for i, r in enumerate(results):

        col1, col2 = st.columns([1, 4])

        with col1:
            if i == 0:
                st.markdown("## 🥇")
            elif i == 1:
                st.markdown("## 🥈")
            elif i == 2:
                st.markdown("## 🥉")
            else:
                st.markdown(f"## #{i+1}")

        with col2:
            st.markdown(f"### 👤 {r['candidate_name']}")
            st.caption(f"📄 {r['file_name']}")
            st.write(f"📧 {r['email']}")

            st.write(
                f"🛠 Matched Skills: {', '.join(r['matched_skills']) if r['matched_skills'] else 'None'}"
            )

            st.progress(int(r["score"]))
            st.write(f"**JD Match:** {r['score']:.2f}%")

        st.divider()

    # ── DOWNLOAD CSV ───────────────────────────
    df = pd.DataFrame(results)
    csv = df.to_csv(index=False).encode()

    st.download_button(
        "⬇ Download CSV",
        csv,
        "results.csv",
        "text/csv"
    )

elif analyze and job_desc.strip() == "":
    st.warning("Please provide a job description.")

elif analyze:
    st.warning("Please upload at least one resume.")