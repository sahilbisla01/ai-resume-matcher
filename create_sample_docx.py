"""
Helper script to generate sample_resume.docx from sample_resume.txt
Run: python create_sample_docx.py
"""
import os
import sys

# Add backend directory to path
backend_path = os.path.join(os.path.dirname(__file__), "backend")
sys.path.insert(0, backend_path)

from docx_export import generate_docx_bytes

def main():
    txt_path = os.path.join(os.path.dirname(__file__), "sample_resume.txt")
    docx_path = os.path.join(os.path.dirname(__file__), "sample_resume.docx")
    
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    docx_bytes = generate_docx_bytes(text)
    
    with open(docx_path, "wb") as f:
        f.write(docx_bytes)
        
    print(f"Successfully generated: {docx_path}")

if __name__ == "__main__":
    main()
