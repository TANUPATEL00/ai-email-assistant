import os
import re
import imaplib
import email
from email.header import decode_header
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------
# Streamlit Config
# ------------------------
st.set_page_config(page_title="AI-Powered Communication Assistant", layout="wide")

SUPPORT_TERMS = ['support','query','request','help']

POSITIVE_WORDS = {'thanks','thank you','appreciate','great','awesome','good','love','happy','satisfied','resolved'}
NEGATIVE_WORDS = {'issue','problem','error','cannot','can\'t','unable','blocked','down',
    'frustrated','upset','angry','annoyed','bad','worst','delay','delayed','fail','failed','failure','refund'}
URGENCY_WORDS = {'urgent','immediately','asap','as soon as possible','critical','high priority',
    'cannot access','can\'t access','blocked','down','outage','service down','system down','locked out'}
REQUIREMENT_MARKERS = ['need','request','require','want','unable','cannot','can\'t','help','guide','fix','resolve','reset']

EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE_RE = re.compile(r'(\+?\d[\d\-\s]{7,}\d)')


# ------------------------
# Utility Functions
# ------------------------
def extract_contacts(text):
    emails = list(set(EMAIL_RE.findall(text or '')))
    phones = list(set(PHONE_RE.findall(text or '')))
    return emails, phones

def detect_sentiment(text):
    tl = (text or '').lower()
    pos = any(w in tl for w in POSITIVE_WORDS)
    neg = any(w in tl for w in NEGATIVE_WORDS)
    if pos and not neg: return 'Positive'
    if neg and not pos: return 'Negative'
    if pos and neg: return 'Mixed'
    return 'Neutral'

def detect_urgency(subject, body):
    tl = f"{subject} {body}".lower()
    hits = [w for w in URGENCY_WORDS if w in tl]
    return ('Urgent' if hits else 'Not urgent'), hits

def extract_requirements(body):
    import re as _re
    sentences = _re.split(r'(?<=[\.!?])\s+', body or '')
    key = [s.strip() for s in sentences if any(m in s.lower() for m in REQUIREMENT_MARKERS)]
    return (' '.join(key[:2]))[:500]

def human_name_from_email(addr):
    import re as _re
    local = (addr or '').split('@')[0]
    name = _re.sub(r'[\.\_\-\+]+', ' ', local).strip().title()
    return name if name else "there"

def generate_reply_rule_based(sender, subject, body, sentiment, priority, urgency_hits, req_summary):
    name = human_name_from_email(sender)
    empathy = ""
    if sentiment in ['Negative','Mixed'] or priority == 'Urgent':
        empathy = " I’m sorry for the trouble you’re facing—we’ll resolve this quickly."
    urgency_line = ""
    if priority == 'Urgent':
        urgency_line = " This has been marked as *high priority*."
    closing = "\n\nWe’ll keep you updated.\n\nBest regards,\nSupport Team"
    return (
        f"Subject: Re: {subject}\n\n"
        f"Hi {name},\n\n"
        f"Thanks for reaching out.{empathy}{urgency_line}\n\n"
        f"Quick recap: {body[:100]}...\n\n"
        f"{'Key details: ' + req_summary if req_summary else ''}\n\n"
        f"{closing}"
    )


def prepare_dataframe(raw_rows):
    df = pd.DataFrame(raw_rows)
    if df.empty:
        return df
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
    if not out.empty:
        order = {'Urgent':0,'Not urgent':1}
        out['p_rank'] = out['priority'].map(order)
        out = out.sort_values(['p_rank','sent_date'], ascending=[True, False]).drop(columns=['p_rank'])
    return out


# ------------------------
# Streamlit App
# ------------------------
def main():
    st.title("📬 AI-Powered Communication Assistant")
    st.caption("Filter, prioritize, extract info, and draft replies")

    # Load demo CSV
    try:
        df_demo = pd.read_csv("sample_data.csv")
        raw_rows = df_demo.to_dict(orient='records')
    except Exception as e:
        st.error("sample_data.csv not found. Please add it in repo root.")
        return

    df = prepare_dataframe(raw_rows)

    if df.empty:
        st.warning("No support emails found in data.")
        return

    # Analytics
    st.subheader("Analytics & Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total filtered", len(df))
    col2.metric("Urgent", int((df['priority']=="Urgent").sum()))
    col3.metric("Negative", int((df['sentiment']=="Negative").sum()))

    # Charts
    fig1 = plt.figure()
    df['sentiment'].value_counts().plot(kind='bar')
    plt.title("Sentiment Distribution")
    st.pyplot(fig1)

    fig2 = plt.figure()
    df['priority'].value_counts().plot(kind='bar')
    plt.title("Priority Distribution")
    st.pyplot(fig2)

    # Emails table
    st.subheader("📌 Filtered Support Emails")
    st.dataframe(df[['sender','subject','sent_date','sentiment','priority','urgency_keywords','requirements_summary']], use_container_width=True)

    # Draft response
    st.subheader("✍️ AI Draft Responses")
    selected = st.selectbox("Pick an email", df.index, format_func=lambda i: f"{df.loc[i,'sender']} — {df.loc[i,'subject']}")
    row = df.loc[selected]

    reply = generate_reply_rule_based(
        sender=row['sender'],
        subject=row['subject'],
        body=row['body'],
        sentiment=row['sentiment'],
        priority=row['priority'],
        urgency_hits=[x.strip() for x in row['urgency_keywords'].split(',') if x.strip()],
        req_summary=row['requirements_summary']
    )

    edited = st.text_area("Draft reply (editable)", value=reply, height=250)
    st.download_button("Download draft as .txt", data=edited, file_name="draft_reply.txt")


if __name__ == "__main__":
    main()
