"""
screening_utils.py
------------------
Core functions for the AI Resume Screening & Candidate Ranking System.

This version is DEPLOYMENT-SAFE: it needs no internet downloads at startup,
so it will never hang while loading on a cloud host.

Pipeline:
    1. extract_text()      -> read raw text from PDF / DOCX / TXT files
    2. preprocess()        -> clean, tokenize, remove stopwords, stem
    3. rank_candidates()   -> TF-IDF vectorize + cosine similarity + sort
"""

import re

from nltk.stem import PorterStemmer  # pure-Python, needs NO downloaded data

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


# Built once and reused. Neither of these requires an internet download.
STOP_WORDS = set(ENGLISH_STOP_WORDS)   # scikit-learn's built-in English stopwords
STEMMER = PorterStemmer()              # reduces words to a common root form


# ---------------------------------------------------------------------------
# 1. TEXT EXTRACTION  --  turn a resume file into a plain string of text
# ---------------------------------------------------------------------------
def extract_text(file_obj, filename):
    """Read text out of a resume file (PDF, DOCX, or TXT)."""
    name = filename.lower()

    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(file_obj)
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if name.endswith(".docx"):
        import docx
        document = docx.Document(file_obj)
        return "\n".join(p.text for p in document.paragraphs)

    if name.endswith(".txt"):
        if isinstance(file_obj, str):                      # a file PATH
            with open(file_obj, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        data = file_obj.read()                             # an uploaded object
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="ignore")
        return data

    raise ValueError(f"Unsupported file type: {filename} (use PDF, DOCX, or TXT)")


# ---------------------------------------------------------------------------
# 2. NLP PREPROCESSING  --  clean the text so comparisons are meaningful
# ---------------------------------------------------------------------------
def preprocess(text):
    """
    Clean one block of text and return it as a single normalized string.

    Steps:
      1. lowercase everything
      2. keep only letters (this also tokenizes the text)
      3. remove stopwords (common filler words)
      4. stem each word to its root (running -> run, engineering -> engin)
      5. drop very short leftover tokens
    """
    tokens = re.findall(r"[a-z]+", text.lower())

    cleaned = []
    for token in tokens:
        if token in STOP_WORDS:
            continue
        root = STEMMER.stem(token)
        if len(root) > 2:
            cleaned.append(root)

    return " ".join(cleaned)


# ---------------------------------------------------------------------------
# 3. RANKING  --  TF-IDF vectorize, score with cosine similarity, then sort
# ---------------------------------------------------------------------------
def rank_candidates(job_description, resumes):
    """
    Rank resumes against a job description.

    resumes : dict {candidate_name: raw_resume_text}
    returns : list of (name, score) sorted best-match first (score 0..1)
    """
    names = list(resumes.keys())

    clean_jd = preprocess(job_description)
    clean_resumes = [preprocess(resumes[n]) for n in names]

    documents = [clean_jd] + clean_resumes

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    scores = cosine_similarity(jd_vector, resume_vectors)[0]

    return sorted(zip(names, scores), key=lambda pair: pair[1], reverse=True)
