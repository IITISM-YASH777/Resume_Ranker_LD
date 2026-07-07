# AI Resume Screening & Candidate Ranking System

An AI system that reads resumes, compares each one against a job description, and
**ranks candidates** by skill relevance using **TF-IDF + Cosine Similarity** (a
classic, interpretable NLP technique).

Built as an Artificial Intelligence capstone project.

---

## What's in this folder

| File | What it is |
|------|------------|
| `resume_screening.ipynb` | The main Jupyter notebook — the full pipeline explained step by step, already run with all outputs, charts, and the evaluation. **This is the core deliverable.** |
| `app.py` | A Streamlit web app: upload resumes, paste a job description, see a live ranking. **This is the deployment demo.** |
| `screening_utils.py` | The shared "engine" (text extraction, preprocessing, TF-IDF + cosine ranking) used by both the notebook and the app. |
| `sample_data/` | One sample job description + five sample resumes so everything runs out of the box. |
| `requirements.txt` | The Python libraries you need. |

---

## How to run it (step by step)

> You only need to do steps 1–3 once.

### 1. Open a terminal **inside this folder**
Everything must be run from the project folder (so the files can find each other).

### 2. (Recommended) create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac / Linux:
source venv/bin/activate
```

### 3. Install the required libraries
```bash
pip install -r requirements.txt
```
The first time the code runs it will automatically download a few small NLTK
data files (stopwords + lemmatizer). This needs an internet connection once.

### 4a. Run the notebook
```bash
jupyter notebook resume_screening.ipynb
```
Then run the cells top to bottom (Cell ▸ Run All). It's already been executed, so
you can also just read it.

### 4b. Run the web app (the demo)
```bash
streamlit run app.py
```
Your browser opens automatically. Paste a job description, upload resumes from the
`sample_data/` folder (or your own PDF/DOCX/TXT files), and click
**Rank Candidates**.

---

## How it works (the 30-second version)

1. **Extract** the text from each resume (PDF / DOCX / TXT).
2. **Preprocess** it: lowercase, remove punctuation and stopwords, lemmatize.
3. **TF-IDF**: turn each document into a numeric vector where distinctive,
   important words get higher weights.
4. **Cosine similarity**: measure how aligned each resume vector is with the job
   description vector — a score from 0 (no match) to 1 (perfect match).
5. **Rank**: sort candidates by that score, best first.

## Example result (from the sample data)

For a *Machine Learning Engineer* job description:

| Rank | Candidate | Match |
|------|-----------|-------|
| 1 | Aisha (ML Engineer) | ~70% |
| 2 | Karan (AI Researcher) | ~49% |
| 3 | Priya (Software Engineer) | ~24% |
| 4 | Rahul (Data Analyst) | ~16% |
| 5 | Sneha (Marketing Manager) | ~1% |

The ranking matches human intuition: technical ML candidates rise to the top, the
unrelated profile sinks to the bottom.

## Honest limitations (discussed in the notebook)

TF-IDF + cosine similarity matches **keywords, not meaning**. It doesn't know
that "ML" = "machine learning" or understand context like "no experience in
Python". The notebook explains how semantic embeddings (e.g. Sentence-BERT) would
improve this — a good "future work" talking point.

---

## Tech stack
Python · scikit-learn · NLTK · pandas · matplotlib · pypdf · python-docx · Streamlit
