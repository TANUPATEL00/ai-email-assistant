
import os
import re
import imaplib
import email
from email.header import decode_header
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------
# Config
# ------------------------
st.set_page_config(page_title="AI-Powered Communication Assistant", layout="wide")

SUPPORT_TERMS = ['support','query','request','help']

POSITIVE_WORDS = {
    'thanks','thank you','appreciate','great','awesome','good','love','happy','satisfied','resolved'
}
NEGATIVE_WORDS = {
    'issue','problem','error','cannot','can\'t','unable','blocked','down',
    'frustrated','upset','angry','annoyed','bad','worst','delay','delayed','fail','failed','failure','refund'
}
URGENCY_WORDS = {
    'urgent','immediately','asap','as soon as possible','critical','high priority',
    'cannot access','can\'t access','blocked','down','outage','service down','system down','locked out'
}
REQUIREMENT_MARKERS = ['need','request','require','want','unable','cannot','can\'t','help','guide','fix','resolve','reset']

EMAIL_RE = re.compile(r'[\\w\\.-]+@[\\w\\.-]+\\.\\w+')
PHONE_RE = re.compile(r'(\\+?\\d[\\d\\-\\s]{7,}\\d)')

def extract_contacts(text):
    emails = list(set(EMAIL_RE.findall(text or '')))
    phones = list(set(PHONE_RE.findall(text or '')))
    return emails, phones

def find_markers(text, vocab):
    tl = (text or '').lower()
    return sorted({w for w in vocab if w in tl})

def detect_sentiment(text):
    tl = (text or '').lower()
    pos = any(w in tl for w in POSITIVE_WORDS)
    neg = any(w in tl for w in NEGATIVE_WORDS)
    if pos and not neg:
        return 'Positive'
    if neg and not pos:
        return 'Negative'
    if pos and neg:
        return 'Mixed'
    return 'Neutral'

def detect_urgency(subject, body):
    hits = find_markers(f"{subject} {body}", URGENCY_WORDS)
    return ('Urgent' if hits else 'Not urgent'), hits

def extract_requirements(body):
    import re as _re
    sentences = _re.split(r'(?<=[\\.!?])\\s+', body or '')
    key = [s.strip() for s in sentences if any(m in s.lower() for m in REQUIREMENT_MARKERS)]
    return (' '.join(key[:2]))[:500]

def human_name_from_email(addr):
    import re as _re
    local = (addr or '').split('@')[0]
    name = _re.sub(r'[\\.\\_\\-\\+]+', ' ', local).strip().title()
    return name if name else "there"

def generate_reply_rule_based(sender, subject, body, sentiment, priority, urgency_hits, req_summary):
    name = human_name_from_email(sender)
    first_sentence = (body or '').split('.', 1)[0]
    words = first_sentence.split()
    short_sum = ' '.join(words[:20]) + ('…' if len(words) > 20 else '')
    empathy = ""
    if sentiment in ['Negative','Mixed'] or priority == 'Urgent':
        empathy = " I’m sorry for the trouble you’re facing—we understand how disruptive this can be, and we’ll work to resolve it quickly."
    urgency_line = ""
    if priority == 'Urgent':
        urgency_line = " I’ve marked this as *high priority* based on cues like: " + ', '.join(urgency_hits) + "."
    steps = (
        "Here’s what we’ll do next:\\n"
        "1) Verify your account and recent activity.\\n"
        "2) Reproduce the issue using the details you shared.\\n"
        "3) Apply the appropriate fix or share a workaround immediately.\\n\\n"
        "If possible, please share:\\n"
        "- A screenshot of the error (if any)\\n"
        "- The exact time and steps just before the issue\\n"
        "- Your app version / browser & OS\\n"
    )
    closing = (
        "\\nWe’ll keep you updated until this is fully resolved. "
        "Thanks for your patience.\\n\\n"
        "Best regards,\\nSupport Team"
    )
    return (
        f"Subject: Re: {subject}\\n\\n"
        f"Hi {name},\\n\\n"
        f"Thanks for reaching out.{empathy}{urgency_line}\\n\\n"
        f"Quick recap of your request: {short_sum}\\n\\n"
        f"{'Key details we captured: ' + req_summary if req_summary else ''}\\n\\n"
        f"{steps}{closing}"
    )

def try_generate_with_openai(prompt):
    # Optional: uses OpenAI if OPENAI_API_KEY is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You write concise, empathetic, professional support replies."},
                {"role":"user","content": prompt}
            ],
            temperature=0.3,
            max_tokens=350
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return None

def fetch_emails_via_imap():
    # Optional IMAP fetch. If IMAP_* env vars are not set, returns empty list.
    host = os.getenv("IMAP_HOST")
    user = os.getenv("IMAP_USER")
    password = os.getenv("IMAP_PASSWORD")
    folder = os.getenv("IMAP_FOLDER", "INBOX")
    limit = int(os.getenv("IMAP_LIMIT", "50"))
    if not host or not user or not password:
        return []
    msgs = []
    imap = imaplib.IMAP4_SSL(host)
    imap.login(user, password)
    imap.select(folder)
    status, data = imap.search(None, "ALL")
    ids = data[0].split()[-limit:]
    for i in ids:
        res, msg_data = imap.fetch(i, "(RFC822)")
        if res != "OK": 
            continue
        msg = email.message_from_bytes(msg_data[0][1])
        def decode(s):
            if not s: return ""
            parts = decode_header(s)
            txt = ""
            for p, enc in parts:
                if isinstance(p, bytes):
                    txt += p.decode(enc or "utf-8", errors="ignore")
                else:
                    txt += p
            return txt
        subject = decode(msg.get("Subject"))
        from_ = decode(msg.get("From"))
        addr = email.utils.parseaddr(from_)[1]
        date_ = msg.get("Date")
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                disp = str(part.get("Content-Disposition"))
                if ctype == "text/plain" and "attachment" not in disp:
                    body = part.get_payload(decode=True).decode(errors="ignore")
                    break
        else:
            body = msg.get_payload(decode=True)
            body = body.decode(errors="ignore") if isinstance(body, bytes) else (body or "")
        msgs.append({"sender": addr, "subject": subject, "body": body, "sent_date": pd.to_datetime(date_, errors='coerce')})
    imap.logout()
    return msgs

def prepare_dataframe(raw_rows):
    df = pd.DataFrame(raw_rows)
    df['sent_date'] = pd.to_datetime(df['sent_date'], errors='coerce')
    mask = df['subject'].fillna('').str.contains('|'.join(SUPPORT_TERMS), case=False, na=False)
    df = df[mask].copy()
    rows = []
    for _, r in df.iterrows():
        subj, body = r.get('subject',''), r.get('body','')
        sentiment = detect_sentiment(subj + " " + body)
        priority, urgency_hits = detect_urgency(subj, body)
        req = extract_requirements(body)
        emails_in_body, phones_in_body = extract_contacts(body)
        rows.append({
            "sender": r['sender'],
            "subject": subj,
            "body": body,
            "sent_date": r['sent_date'],
            "sentiment": sentiment,
            "priority": priority,
            "urgency_keywords": ', '.join(urgency_hits),
            "contacts_found_emails": ', '.join(emails_in_body),
            "contacts_found_phones": ', '.join(phones_in_body),
            "requirements_summary": req
        })
    out = pd.DataFrame(rows)
    order = {'Urgent':0,'Not urgent':1}
    out['p_rank'] = out['priority'].map(order)
    out = out.sort_values(['p_rank','sent_date'], ascending=[True, False]).drop(columns=['p_rank'])
    return out

def main():
    st.title("📬 AI-Powered Communication Assistant")
    st.caption("Filter, prioritize, extract info, and draft replies — end to end")

    # Sidebar: data source
    mode = st.sidebar.selectbox("Data source", ["Demo (CSV)", "IMAP (Inbox)"])
    if mode == "Demo (CSV)":
        demo_path = st.sidebar.text_input("CSV path", "sample_data.csv")
        if st.sidebar.button("Load CSV"):
            st.session_state['data_source'] = ('csv', demo_path)
    else:
        st.sidebar.write("Uses IMAP_* environment variables")
        if st.sidebar.button("Fetch Emails"):
            st.session_state['data_source'] = ('imap', None)

    raw_rows = []
    if st.session_state.get('data_source') == ('imap', None):
        raw_rows = fetch_emails_via_imap()
        if not raw_rows:
            st.warning("No emails fetched or IMAP not configured. Falling back to Demo.")
    if not raw_rows:
        # demo load
        path = st.session_state.get('data_source', ('csv','sample_data.csv'))[1] or 'sample_data.csv'
        try:
            df_demo = pd.read_csv(path)
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")
            st.stop()
        raw_rows = df_demo.to_dict(orient='records')

    df = prepare_dataframe(raw_rows)

    # Analytics
    st.subheader("Analytics & Stats")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total filtered", len(df))
    col2.metric("Urgent", int((df['priority']=="Urgent").sum()))
    col3.metric("Not urgent", int((df['priority']=="Not urgent").sum()))
    col4.metric("Negative", int((df['sentiment']=="Negative").sum()))
    col5.metric("Last 24h", int((df['sent_date'] >= (pd.Timestamp.now() - pd.Timedelta('24h'))).sum()))

    # Bar chart for sentiment & priority
    fig1 = plt.figure()
    df['sentiment'].value_counts().plot(kind='bar')
    plt.title("Sentiment Distribution")
    st.pyplot(fig1)

    fig2 = plt.figure()
    df['priority'].value_counts().plot(kind='bar')
    plt.title("Priority Distribution")
    st.pyplot(fig2)

    st.subheader("📌 Filtered Support Emails (Priority-queued)")
    st.dataframe(df[['sender','subject','sent_date','sentiment','priority','urgency_keywords','contacts_found_emails','contacts_found_phones','requirements_summary']], use_container_width=True)

    st.subheader("✍️ AI Draft Responses")
    selected = st.selectbox("Pick an email to draft a reply", df.index, format_func=lambda i: f"{df.loc[i,'sender']} — {df.loc[i,'subject']}")
    row = df.loc[selected]
    # Try LLM (OpenAI) with RAG-like prompt, else use rule-based
    kb_text = ""
    for kb_file in sorted([f for f in os.listdir('knowledge_base') if f.endswith(('.md','.txt'))] if os.path.isdir('knowledge_base') else []):
        try:
            kb_text += open(os.path.join('knowledge_base', kb_file), 'r', encoding='utf-8').read() + "\n\n"
        except Exception:
            pass

    rag_prompt = f"""
You are a support specialist. Write a concise, empathetic, professional reply.
Use this knowledge base when useful:\n---\n{kb_text[:4000]}\n---
Email:
From: {row['sender']}
Subject: {row['subject']}
Body: {row['body']}

Provide a short recap, acknowledge tone (if negative/frustrated), propose next steps,
and request any missing details that would help resolution. Keep it under 180 words.
"""

    reply = try_generate_with_openai(rag_prompt)
    if reply is None:
        reply = generate_reply_rule_based(
            sender=row['sender'],
            subject=row['subject'],
            body=row['body'],
            sentiment=row['sentiment'],
            priority=row['priority'],
            urgency_hits=[x.strip() for x in row['urgency_keywords'].split(',') if x.strip()],
            req_summary=row['requirements_summary']
        )

    edited = st.text_area("Draft (you can edit before sending)", value=reply, height=250)
    st.code(edited)

    st.download_button("Download draft as .txt", data=edited, file_name="draft_reply.txt")

    st.info("To enable IMAP and OpenAI, set environment variables (see README).")

if __name__ == "__main__":
    main()
