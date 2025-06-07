import streamlit as st
import json
import zipfile
import os
from agent_logic import run_customer_support, chunk_with_overlap

def load_kb_from_zip(zip_path, json_filename):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract(json_filename, path=".")
    with open(json_filename, 'r', encoding='utf-8') as f:
        kb_data = json.load(f)
    os.remove(json_filename)
    return kb_data

@st.cache_data(show_spinner="Loading knowledge base...")
def load_kb_chunks():
    kb_data = load_kb_from_zip("knowledge_base.zip", "knowledge_base.json")
    return chunk_with_overlap(kb_data)

kb_chunks = load_kb_chunks()


# -------------------- FAQs --------------------
FAQS = {
    "What are your business hours?": "Our business hours are from 9 AM to 7 PM, Monday to Friday. After hours, you can reach us at customer@support.ai.",
    "How do I cancel or change my subscription?": "You can manage or cancel your subscription anytime via your Account Dashboard. We’d love your feedback if you're leaving, so we can improve!",
    "Why was I charged twice?": "We’re sorry for the confusion. Sometimes, duplicate charges occur due to payment retries. Please email *billing@support.ai* with your invoice ID, and we’ll resolve it immediately.",
    "Do you offer discounts or promotions?": "Yes! We offer up to 30% OFF for students. Just upload valid proof of student status to avail the offer.",
    "How do I update my payment method?": "Go to your *Account Settings → Billing* section to securely update your card or payment method at any time.",
    "The website isn’t working. What should I do?": "Try clearing your browser cache or switching to another browser. If the issue persists, contact us at *support@support.ai* with details/screenshots."
}

# -------------------- Layout Setup --------------------


# Custom CSS for improved chat, background, and sidebar
st.markdown("""
    <style>
        body, .stApp {
            background: #011f3a !important;
        }
        .main-title {
            font-size: 2.6rem;
            font-weight: 900;
            letter-spacing: 2px;
            margin-bottom: 0.3em;
            text-align: center;
            background: linear-gradient(90deg, #e0f7fa 0%, #b2ebf2 50%, #ffffff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-fill-color: transparent;
        }
        .subtitle {
            font-size: 1.2rem;
            font-style: italic;
            color: #e0f7fa;
            text-align: center;
            margin-bottom: 1.2em;
        }
        .sidebar-header {
            background: linear-gradient(90deg, #e0f7fa 60%, #b2ebf2 100%);
            padding: 18px 10px 10px 10px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 18px;
            box-shadow: 0 2px 8px #b2ebf2;
        }
        .sidebar-header h3 {
            color: #111;
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 0.2em;
            letter-spacing: 1px;
        }
        .sidebar-header p {
            color: #222;
            font-size: 1rem;
            font-style: italic;
        }
        .bot, .user {
            padding: 12px 18px;
            margin: 10px 0;
            border-radius: 22px;
            font-size: 1.09rem;
            max-width: 75%;
            line-height: 1.5;
            display: inline-block;
            clear: both;
            word-break: break-word;
            color: #111;
        }
        .bot {
            background-color: #e3f2fd;
            float: left;
            border-bottom-left-radius: 4px;
            border-top-left-radius: 22px;
            border-top-right-radius: 22px;
            border-bottom-right-radius: 22px;
            box-shadow: 0 2px 8px #b3e5fc44;
        }
        .user {
            background-color: #c8e6c9;
            float: right;
            border-bottom-right-radius: 4px;
            border-top-left-radius: 22px;
            border-top-right-radius: 22px;
            border-bottom-left-radius: 22px;
            box-shadow: 0 2px 8px #a5d6a7aa;
        }
        .stTextInput>div>div>input {
            color: #111 !important;
            background: #f9f9f9 !important;
        }
        header, footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <div class="sidebar-header">
        <h3>💬 Support Agent</h3>
        <p>Chat in a modern, WhatsApp-style interface</p>
    </div>
    """,
    unsafe_allow_html=True,
)
page = st.sidebar.radio("📚 Navigation", ["Chat", "FAQs"])

# -------------------- Main Title and Description --------------------
st.markdown('<div class="main-title">Supportify: AI Customer Support Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your 24/7 intelligent assistant for instant, reliable, and human-like customer support.<br><i>Ask anything — Supportify is always here to help!</i></div>', unsafe_allow_html=True)

# -------------------- WhatsApp-style Chat Page --------------------
if page == "Chat":
    st.subheader("📱 Customer Support Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [("bot", "👋 Hello! Welcome to Supportify. How can I assist you today?")]

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Type a message...", label_visibility="collapsed", max_chars=500)
        submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        st.session_state.chat_history.append(("user", user_input.strip()))

        # Pass both user_input and kb_chunks to the agent logic
        result = run_customer_support(user_input.strip(), kb_chunks)
        reply = result.get("response", "Sorry, I couldn't understand that. Please rephrase or check our FAQs.")
        cat = result.get("category", "")
        sent = result.get("sentiment", "")

        # Show category/sentiment (optional, comment out if not wanted)
        meta = f"<span style='font-size:0.96em;color:#0097a7;'>[Category: <b>{cat}</b> | Sentiment: <b>{sent}</b>]</span><br/>"
        st.session_state.chat_history.append(("bot", meta + reply))

    # Render chat history in WhatsApp style
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for sender, msg in st.session_state.chat_history:
        css_class = "user" if sender == "user" else "bot"
        st.markdown(f'<div class="{css_class}">{msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- FAQs Page --------------------
elif page == "FAQs":
    st.subheader("📄 Frequently Asked Questions")

    st.markdown(
        """
        <style>
            .faq-container {
                background-color: #0a2740;
                padding: 24px;
                border-radius: 14px;
                border: 1px solid #b2ebf2;
                max-width: 700px;
                margin: auto;
                font-family: 'Segoe UI', sans-serif;
            }
            .faq-question {
                font-weight: bold;
                font-size: 18px;
                margin-top: 20px;
                color: #00eaff;
            }
            .faq-answer {
                font-style: italic;
                margin-bottom: 12px;
                color: #fff;
                font-size: 15px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="faq-container">', unsafe_allow_html=True)
    for question, answer in FAQS.items():
        st.markdown(f'<div class="faq-question">Q: {question}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="faq-answer">A: {answer}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
