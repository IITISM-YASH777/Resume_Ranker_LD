"""
screening_utils.py
------------------
Core functions for the AI Resume Screening & Candidate Ranking System.

This single file holds the "engine" of the project so that BOTH the Jupyter
notebook and the Streamlit app can import and reuse the exact same logic.
Keeping shared code in one place is good practice and avoids copy-paste bugs.

Pipeline implemented here:
    1. extract_text()      -> read raw text from PDF / DOCX / TXT files
    2. preprocess()        -> clean, tokenize, remove stopwords, lemmatize
    3. rank_candidates()   -> TF-IDF vectorize + cosine similarity + sort

Author: (your name)
"""

import re
import io

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# 0. One-time NLTK data download (safe to call repeatedly; it's cached)
# ---------------------------------------------------------------------------
def ensure_nltk_data():
    """Download the small NLTK data files we need, if not already present."""
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


ensure_nltk_data()

# Build these once at import time (they are reused for every document).
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# 1. TEXT EXTRACTION  --  turn a resume file into a plain string of text
# ---------------------------------------------------------------------------
def extract_text(file_obj, filename):
    """
    Read text out of a resume file.

    Parameters
    ----------
    file_obj : a file path (str) OR a file-like object (e.g. from an upload)
    filename : the file's name, used only to detect the extension

    Returns
    -------
    str : the raw extracted text (may be messy; that's fine, we clean it next)
    """
    name = filename.lower()

    # --- PDF ---------------------------------------------------------------
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(file_obj)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    # --- DOCX (Microsoft Word) --------------------------------------------
    if name.endswith(".docx"):
        import docx
        document = docx.Document(file_obj)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    # --- TXT (plain text) --------------------------------------------------
    if name.endswith(".txt"):
        if isinstance(file_obj, str):           # it's a file PATH
            with open(file_obj, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        # it's an uploaded file-like object (bytes)
        data = file_obj.read()
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

    Steps (this is the "NLP preprocessing" your project requires):
      1. lowercase everything            -> "Python" and "python" become equal
      2. keep only letters (regex)       -> drops numbers, punctuation, symbols
                                            (this also does the tokenizing for us,
                                             which avoids fragile NLTK tokenizers)
      3. remove stopwords                -> drops "the", "and", "is", ...
      4. lemmatize                       -> "running" -> "run", "managers" -> "manager"
      5. drop very short tokens          -> removes leftover noise like "a", "x"
    """
    # 1 + 2: lowercase, then grab runs of letters only -> a list of word tokens
    tokens = re.findall(r"[a-z]+", text.lower())

    cleaned = []
    for token in tokens:
        if token in STOP_WORDS:        # 3: skip common filler words
            continue
        lemma = LEMMATIZER.lemmatize(token)   # 4: reduce to base form
        if len(lemma) > 2:             # 5: ignore 1-2 letter leftovers
            cleaned.append(lemma)

    # Join back into one string -- TfidfVectorizer expects strings, not lists
    return " ".join(cleaned)


# ---------------------------------------------------------------------------
# 3. RANKING  --  TF-IDF vectorize, score with cosine similarity, then sort
# ---------------------------------------------------------------------------
def rank_candidates(job_description, resumes):
    """
    Rank resumes against a job description.

    Parameters
    ----------
    job_description : str   -- the raw job description text
    resumes         : dict  -- {candidate_name: raw_resume_text}

    Returns
    -------
    list of (name, score) tuples, sorted from best match to worst.
    score is the cosine similarity, a number between 0 and 1.
    """
    names = list(resumes.keys())

    # Clean the JD and every resume with the SAME preprocessing function.
    clean_jd = preprocess(job_description)
    clean_resumes = [preprocess(resumes[name]) for name in names]

    # Put the JD first, then all resumes, into one list of documents.
    # We fit the vectorizer on ALL of them so they share one vocabulary.
    documents = [clean_jd] + clean_resumes

    # --- TF-IDF: turn each document into a numeric vector ------------------
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Row 0 is the JD; rows 1..N are the resumes.
    jd_vector = tfidf_matrix[0:1]        # shape (1, vocab)
    resume_vectors = tfidf_matrix[1:]    # shape (N, vocab)

    # --- Cosine similarity: how aligned is each resume with the JD? --------
    # Returns one similarity score per resume, each between 0 and 1.
    scores = cosine_similarity(jd_vector, resume_vectors)[0]

    # --- Rank: pair names with scores and sort high -> low -----------------
    ranked = sorted(zip(names, scores), key=lambda pair: pair[1], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# 4. CANDIDATE NAME  --  pull the applicant's real name out of the resume
# ---------------------------------------------------------------------------
# A name line is 1-4 capitalized words, no digits, no "@", not too long.
_NAME_LINE_RE = re.compile(r"^[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,3}$")

# Lines containing these words are headers/contact info, never a name.
_NOT_A_NAME = (
    "resume", "curriculum vitae", "cv", "profile", "objective", "summary",
    "address", "email", "phone", "contact", "linkedin", "github", "portfolio",
)


def extract_candidate_name(resume_text, fallback):
    """
    Guess the applicant's real name from the top of their resume so the UI
    can show "Aisha Khan" instead of a filename like "DOC3449".

    Falls back to `fallback` (typically the uploaded filename without its
    extension) if nothing in the first few lines looks like a name.
    """
    for line in resume_text.splitlines()[:15]:
        line = line.strip()
        if not line or len(line) > 40 or "@" in line:
            continue
        if any(char.isdigit() for char in line):
            continue
        if any(bad in line.lower() for bad in _NOT_A_NAME):
            continue
        if _NAME_LINE_RE.match(line):
            return line.title() if line.isupper() else line
    return fallback


# ---------------------------------------------------------------------------
# 5. SKILL MATCHING  --  which key terms from the JD actually show up
# ---------------------------------------------------------------------------
# Generic words that survive stopword-removal but aren't real "skills".
_GENERIC_TERMS = {
    "experience", "strong", "ideal", "candidate", "looking", "develop",
    "developing", "design", "designing", "build", "building",
    "responsibility", "responsibilities", "required", "requirement",
    "skill", "skills", "work", "working", "year", "years", "role", "team",
    "ability", "knowledge", "understanding", "proficient", "proficiency",
    "familiar", "familiarity", "excellent", "good", "hand", "hands",
    "include", "including", "various", "job", "title", "company", "join",
    "description", "preferred", "plus", "must", "using", "use", "task",
    "tasks", "collaborate", "collaborating", "platform", "platforms",
}


def get_top_keywords(text, top_n=12):
    """
    Rank the most important terms in a document (typically the job
    description) using TF-IDF weight, so we get a short list of the
    proficiencies the role actually cares about.
    """
    clean = preprocess(text)
    if not clean.strip():
        return []

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([clean])
    scores = matrix.toarray()[0]
    terms = vectorizer.get_feature_names_out()

    ranked = sorted(zip(terms, scores), key=lambda pair: pair[1], reverse=True)
    keywords = [term for term, score in ranked if score > 0 and term not in _GENERIC_TERMS]
    return keywords[:top_n]


def match_skills(jd_keywords, resume_text):
    """
    Split the JD's key terms into what this resume DOES and DOESN'T cover.

    Returns (matched, missing) -- two lists of keywords, in the same
    order as jd_keywords.
    """
    resume_tokens = set(preprocess(resume_text).split())
    matched = [kw for kw in jd_keywords if kw in resume_tokens]
    missing = [kw for kw in jd_keywords if kw not in resume_tokens]
    return matched, missing
