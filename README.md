# Intelligent Lexical Analyzer Using Machine Learning

A small runnable educational prototype that tokenizes code-like input and predicts token classes using a supervised machine-learning model with neighboring-token context features.

## Features
- Token extraction using a regular-expression lexer
- Context features: previous token, next token, token shape, and length
- Decision Tree classifier from scikit-learn
- Streamlit interface with token table and class summary

## Run locally
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## GitHub submission
Create a new GitHub repository, then run from this folder:
```bash
git init
git add .
git commit -m "Add intelligent lexical analyzer"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```
Submit the repository URL in the form's **Code: GitHub link** field.

## Important note
This is a demonstration-scale project. The classifier uses a small illustrative training set and does not replace a complete compiler lexer/parser. Explain that the project demonstrates how neighboring-token context can be incorporated into ML-based token classification.
