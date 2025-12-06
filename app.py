
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(page_title="Hire3x Smart Rank", page_icon="🧠", layout="wide")
st.title("🧠 Hire3x Smart Rank – Candidate Matching System")
st.write("Automatically rank candidates for multiple jobs based on skills and experience.")

# -------------------------------
# Upload Job CSV
# -------------------------------
st.header("📁 Upload Job CSV")
st.write("CSV must contain: `Job Title`, `Required Skills`, `Minimum Experience`")
job_file = st.file_uploader("Upload Job CSV", type=["csv"], key="job_csv")

# -------------------------------
# Upload Candidate CSV
# -------------------------------
st.header("📁 Upload Candidate CSV")
st.write("CSV must contain: `Name`, `Skills`, `Experience`")
candidate_file = st.file_uploader("Upload Candidate CSV", type=["csv"], key="candidate_csv")

# -------------------------------
# Matching Logic
# -------------------------------
def calculate_match(job_skills, job_exp, candidates_df):
    job_skills_text = job_skills.lower()
    results = []
    
    # Convert job_exp to int
    try:
        job_exp = int(job_exp)
    except (ValueError, TypeError):
        job_exp = 0

    for _, row in candidates_df.iterrows():
        candidate_skills = str(row["Skills"]).lower()
        experience = row.get("Experience", 0)
        
        # Convert experience to float
        try:
            experience = float(experience)
        except (ValueError, TypeError):
            experience = 0

        # Skill Match using cosine similarity
        vectorizer = CountVectorizer().fit_transform([job_skills_text, candidate_skills])
        vectors = vectorizer.toarray()
        cosine_sim = cosine_similarity(vectors)[0][1]

        # Experience score
        exp_score = min(experience / job_exp, 1.0) if job_exp > 0 else 1.0

        # Final weighted score
        final_score = (cosine_sim * 0.7 + exp_score * 0.3) * 100

        results.append({
            "Name": row["Name"],
            "Skills": row["Skills"],
            "Experience": experience,
            "Match %": round(final_score, 2)
        })

    ranked = pd.DataFrame(results).sort_values(by="Match %", ascending=False).reset_index(drop=True)
    return ranked

# -------------------------------
# Skill Filtering Functions
# -------------------------------
def has_any_skill(candidate_skills, required_skills):
    candidate_set = set([s.strip().lower() for s in candidate_skills.split(",")])
    required_set = set([s.strip().lower() for s in required_skills.split(",")])
    return not required_set.isdisjoint(candidate_set)

def has_all_skills(candidate_skills, required_skills):
    candidate_set = set([s.strip().lower() for s in candidate_skills.split(",")])
    required_set = set([s.strip().lower() for s in required_skills.split(",")])
    return required_set.issubset(candidate_set)

# Highlight skills
def highlight_skills(candidate_skills, required_skills):
    candidate_list = [s.strip() for s in candidate_skills.split(",")]
    required_set = set([s.strip().lower() for s in required_skills.split(",")])
    highlighted = [
        f"**{skill}**" if skill.strip().lower() in required_set else skill
        for skill in candidate_list
    ]
    return ", ".join(highlighted)

# -------------------------------
# Process Uploaded CSVs
# -------------------------------
if job_file and candidate_file:
    jobs_df = pd.read_csv(job_file)
    candidates_df = pd.read_csv(candidate_file)

    st.header("🏆 Ranked Candidates per Job")

    # Skill filter type
    filter_type = st.radio(
        "Skill Filter Type",
        ["ANY required skill", "ALL required skills"],
        index=0
    )

    for _, job in jobs_df.iterrows():
        st.subheader(f"Job: {job['Job Title']}")
        required_skills = job["Required Skills"]
        min_exp = job["Minimum Experience"]

        # Apply skill filter
        if filter_type == "ALL required skills":
            filtered_candidates = candidates_df[
                candidates_df["Skills"].apply(lambda x: has_all_skills(str(x), required_skills))
            ]
        else:
            filtered_candidates = candidates_df[
                candidates_df["Skills"].apply(lambda x: has_any_skill(str(x), required_skills))
            ]

        if filtered_candidates.empty:
            st.warning("No candidates match the filter for this job.")
            continue

        ranked_results = calculate_match(required_skills, min_exp, filtered_candidates)
        ranked_results["Skills"] = ranked_results["Skills"].apply(lambda x: highlight_skills(x, required_skills))

        st.success(f"Top candidate: {ranked_results.iloc[0]['Name']} 🎯")
        st.dataframe(ranked_results, use_container_width=True)

        # Bar chart
        fig = px.bar(
            ranked_results,
            x="Name",
            y="Match %",
            color="Match %",
            text="Match %",
            title=f"Candidate Match % for {job['Job Title']}"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Download ranked candidates CSV
        csv = ranked_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Ranked Candidates CSV",
            data=csv,
            file_name=f"ranked_candidates_{job['Job Title'].replace(' ', '_')}.csv",
            mime="text/csv"
        )
else:
    st.info("Please upload both Job and Candidate CSV files to start matching.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("🚀 Hire3x SmartRank – Full Project Version")
