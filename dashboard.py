import streamlit as st
import pandas as pd
import numpy as np
import os
from sentence_transformers import util
import gdown

ART = "artifacts"

# 🔽 Download artifacts from Google Drive
def download_artifacts():
    if not os.path.exists(ART):
        os.makedirs(ART)

    files = {
        "resumes_cleaned.csv": "1ILTXPEazCokj9HXqmQAM_0djEEvThyE8",
# ✅ already added
        "jobs_cleaned.csv": "1HAFU7EK2wyhXpU4ZMXWqsShkNxEtmTCB",
        "resume_embeddings.npy": "1LJIo6iL_B5n2-SzW5ohklMKVA9daM3ug",
        "job_embeddings.npy": "1OTcUUljvqe3wZ2jrvWnewy7otMzhwvy_"
    }

    for filename, file_id in files.items():
        file_path = os.path.join(ART, filename)

        # Download only if file not exists
        if not os.path.exists(file_path):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, file_path, quiet=False)

# 🔽 Load data
@st.cache_data
def load_data():
    resumes = pd.read_csv(os.path.join(ART, "resumes_cleaned.csv"))
    jobs = pd.read_csv(os.path.join(ART, "jobs_cleaned.csv"))
    resume_emb = np.load(os.path.join(ART, "resume_embeddings.npy"))
    job_emb = np.load(os.path.join(ART, "job_embeddings.npy"))
    return resumes, jobs, resume_emb, job_emb

# 🔽 Main app
def main():
    st.set_page_config(page_title="Resume ↔ Job Matcher", layout="wide")
    st.title("💼 Resume ↔ Job Matching Dashboard")

    # 📥 Download required files
    download_artifacts()

    # 📊 Load data
    resumes, jobs, resume_emb, job_emb = load_data()

    # 🎛 Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        job_idx = st.number_input("Select Job Index", min_value=0, max_value=len(jobs)-1, value=0)
        top_k = st.slider("Filter Top K Matches", 1, 50, 10)

    # 🧾 Show selected job
    st.markdown("### 🧾 Selected Job Description")
    st.info(jobs.iloc[job_idx]['raw_text'])

    # 🔍 Compute similarity
    job_vector = job_emb[job_idx:job_idx+1]
    sims = util.cos_sim(job_vector, resume_emb).cpu().numpy().ravel()
    topk = sims.argsort()[-top_k:][::-1]

    # 🎯 Show top matches
    st.markdown("### 🎯 Top Matching Resumes")
    selected_resume = st.radio(
        "Select a Resume ID to View Details:",
        [f"Resume ID {resumes.index[idx]} (Score: {sims[idx]:.3f})" for idx in topk]
    )

    # 📄 Show selected resume
    if selected_resume:
        selected_id = int(selected_resume.split()[2])
        st.markdown("### 📄 Resume Preview")
        st.success(f"Showing details for Resume ID: {selected_id}")
        st.write(resumes.iloc[selected_id]['raw_text'])

        # ✅ Save manual label
        if st.button(f"✅ Mark as Good Match for Job {job_idx}"):
            with open(os.path.join(ART, "manual_labels.csv"), "a", encoding="utf8") as f:
                f.write(f"{job_idx},{selected_id},1\n")
            st.toast("Saved successfully!", icon="💾")

    # 📌 Footer
    st.markdown("---")
    st.caption("💡 Manual labels are saved in artifacts/manual_labels.csv")

# 🚀 Run app
if __name__ == "__main__":
    main()