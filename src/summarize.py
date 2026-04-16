import google.generativeai as genai
import os

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-1.5-flash")

RISK_KEYWORDS = {
    'high': ['shall not', 'prohibited', 'liable', 'penalty', 'terminate', 'damages', 'irrevocable'],
    'medium': ['may not', 'restricted', 'limited', 'subject to', 'condition'],
    'low': ['may', 'optional', 'reasonable', 'mutual']
}

def assess_risk(text: str) -> str:
    text_lower = text.lower()
    for keyword in RISK_KEYWORDS['high']:
        if keyword in text_lower:
            return 'HIGH'
    for keyword in RISK_KEYWORDS['medium']:
        if keyword in text_lower:
            return 'MEDIUM'
    return 'LOW'

def summarize_clause(clause_type: str, text: str) -> dict:
    prompt = f"""You are a legal assistant. Summarize this {clause_type} clause in 1-2 plain English sentences that a non-lawyer can understand. Be concise and direct.

Clause text:
{text[:1000]}

Respond with only the summary, no preamble."""

    try:
        response = gemini.generate_content(prompt)
        summary = response.text.strip()
    except Exception as e:
        summary = f"Could not generate summary: {str(e)}"

    return {
        'clause_type': clause_type,
        'original_text': text,
        'summary': summary,
        'risk_level': assess_risk(text)
    }

def summarize_all(clauses: list) -> list:
    return [summarize_clause(c['clause_type'], c['text']) for c in clauses]