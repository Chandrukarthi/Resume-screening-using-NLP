
skills_list = [
    "python", "sql", "machine learning", "data analysis",
    "excel", "deep learning", "html", "css", "javascript"
]

def extract_skills(text):
    text = text.lower()
    return list(set([skill for skill in skills_list if skill in text]))