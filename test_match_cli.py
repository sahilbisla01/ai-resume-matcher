"""
Quick CLI verification script to test sample_resume.txt against sample_job_description.txt
Run: python test_match_cli.py
"""
import os
import sys

# Add backend directory
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
sys.path.insert(0, backend_dir)

from nlp_extractor import extract_entities, compute_baseline_score

def main():
    jd_file = os.path.join(os.path.dirname(__file__), "sample_job_description.txt")
    resume_file = os.path.join(os.path.dirname(__file__), "sample_resume.txt")

    if not os.path.exists(jd_file) or not os.path.exists(resume_file):
        print("❌ Error: Sample files not found!")
        return

    with open(jd_file, "r", encoding="utf-8") as f:
        jd_text = f.read()

    with open(resume_file, "r", encoding="utf-8") as f:
        resume_text = f.read()

    print("\n" + "=" * 60)
    print("🎯 AI RESUME MATCHER - SAMPLE VERIFICATION TEST")
    print("=" * 60)

    # 1. Entity Extraction
    resume_entities = extract_entities(resume_text)
    jd_entities = extract_entities(jd_text)

    print("\n[1] Extracted Entities:")
    print(f"  • Candidate Experience Detected : {resume_entities.years_experience} years")
    print(f"  • Candidate Degrees Detected    : {resume_entities.degrees}")
    print(f"  • Target JD Requirements        : {jd_entities.years_experience} years exp | Degrees: {jd_entities.degrees}")

    # 2. Baseline Matching
    baseline = compute_baseline_score(resume_text, jd_text)
    print("\n[2] ATS Baseline Match Results:")
    print(f"  • Keyword Match Score : {baseline.keyword_match_score:.1f}%")
    print(f"  • Token Overlap Score : {baseline.overlap_score:.1f}%")
    
    print("\n[3] Skills Breakdown:")
    print(f"  ✅ Matched Skills ({len(baseline.matched_skills)}):")
    print("     " + ", ".join(baseline.matched_skills))
    
    print(f"\n  ⚠️  Missing Skills / Target Gaps ({len(baseline.missing_skills)}):")
    print("     " + ", ".join(baseline.missing_skills))

    print("\n" + "=" * 60)
    print("🎉 Verification completed successfully!")
    print("Next step: Run the web UI using 'streamlit run frontend/streamlit_app.py' or 'run.bat'")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
