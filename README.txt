
# AI-Powered Communication Assistant (Streamlit)

A simple end-to-end assistant that:
- Fetches emails (IMAP optional) or loads a CSV demo
- Filters support-related emails
- Extracts contacts & requirements
- Detects sentiment and urgency and builds a priority queue
- Generates AI replies (OpenAI if configured, else rule-based)
- Shows analytics and a dashboard

## Quickstart (Demo Mode)

1. Create and activate a virtualenv (Python 3.9+ recommended).
2. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Put your sample CSV as `sample_data.csv` (provided) or point to your own file in the sidebar.
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## IMAP (Optional)

Set environment variables before running:

```bash
export IMAP_HOST="imap.gmail.com"
export IMAP_USER="you@example.com"
export IMAP_PASSWORD="your_app_password"
export IMAP_FOLDER="INBOX"     # optional
export IMAP_LIMIT="50"         # optional
```

## OpenAI (Optional)

```bash
export OPENAI_API_KEY="sk-..."
```

The app uses `gpt-4o-mini` to craft contextual, empathetic replies. If not set, it falls back to a rule-based template.

## Knowledge Base (RAG-lite)

Put Markdown/TXT files into `knowledge_base/` to seed replies with product FAQs, policies, SLAs, etc.

## Files

- `app.py` – Streamlit app
- `requirements.txt`
- `sample_data.csv` – demo data
- `knowledge_base/faq.md` (example)
- `knowledge_base/policies.md` (example)

