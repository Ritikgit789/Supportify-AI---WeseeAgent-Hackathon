import json
import re
from typing import Dict, TypedDict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import zipfile


# Load API key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

# --- Load KB from ZIP ---
def load_kb_from_zip(zip_path: str, json_filename: str) -> List[Dict]:
    """Extract JSON from ZIP and load as Python object."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract(json_filename, path=".")
    with open(json_filename, 'r', encoding='utf-8') as f:
        kb_data = json.load(f)
    os.remove(json_filename)  # Clean up extracted file
    return kb_data

# --- Knowledge Base Loading and Chunking ---
def chunk_with_overlap(kb_data, chunk_size=700, chunk_overlap=100):
    # If kb_data is a string, treat it as a file path
    if isinstance(kb_data, str):
        with open(kb_data, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    else:
        # Otherwise, assume it's already loaded JSON data (a list)
        raw_data = kb_data

    chunks = []
    for item in raw_data:
        text = item['text']
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i+chunk_size]
            chunks.append({
                "content": chunk,
                "meta": {"folder": item['folder'], "file": item['file']}
            })
    return chunks


# --- Fast Keyword Search ---
def find_relevant_chunks(query: str, chunks: List[Dict], top_k=5):
    query_words = set(re.findall(r'\w+', query.lower()))
    scored = []
    for chunk in chunks:
        chunk_words = set(re.findall(r'\w+', chunk["content"].lower()))
        score = len(query_words & chunk_words)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in scored[:top_k]]

def format_context(chunks):
    context = []
    for chunk in chunks:
        meta = chunk["meta"]
        context.append(f"Source: {meta['folder']}/{meta['file']}\n{chunk['content']}")
    return "\n\n".join(context)

# --- Agent Workflow ---
class State(TypedDict):
    query: str
    category: str
    sentiment: str
    response: str

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

def categorize(query: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following customer query into one of these categories ONLY: Technical, Billing, General.\n"
        "Return only the category name.\n\nQuery: {query}"
    )
    return (prompt | llm).invoke({"query": query}).content.strip()

def analyze_sentiment(query: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following customer query. Respond with either 'Positive', 'Neutral', or 'Negative'. Query: {query}"
    )
    return (prompt | llm).invoke({"query": query}).content.strip()

def escalate(query: str) -> str:
    return (
        "We're sorry your issue couldn't be resolved automatically.\n"
        "Be patient. A human agent will contact you soon.\n"
        "Or please contact our service number - 0123456789 \n"
        "Our business hours: 9 AM to 6 PM, Monday to Friday.\n"
        "Contact: customersupport@supportai.com"
    )
def handle_technical(state: State) -> State:
    chunks = find_relevant_chunks(state["query"])
    context = format_context(chunks)
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful and friendly technical support agent.\n"
        "Use the following knowledge base info to answer:\n\n{context}\n\n"
        "Query: {query}\n\n"
        "Don't add ** in responses.\n"
        "This is a critical technical issue. We'll connect you to a human agent shortly.\n"
        "Business hours: 9 AM – 6 PM, Mon–Fri. Contact: support@example.com"
    )
    chain = prompt | llm
    return {"response": chain.invoke({"context": context, "query": state["query"]}).content.strip()}

def handle_billing(state: State) -> State:
    chunks = find_relevant_chunks(state["query"])
    context = format_context(chunks)
    prompt = ChatPromptTemplate.from_template(
        "You are a polite billing support agent.\n"
        "Use the following info from our knowledge base:\n\n{context}\n\n"
        "Answer the billing query clearly:\n\nQuery: {query}\n\n"
        "Don't use ** in lines.\n"
        "Handle refunds, duplicate charges, receipts, or payment queries accordingly."
    )
    chain = prompt | llm
    return {"response": chain.invoke({"context": context, "query": state["query"]}).content.strip()}

def handle_general(state: State) -> State:
    chunks = find_relevant_chunks(state["query"])
    context = format_context(chunks)
    prompt = ChatPromptTemplate.from_template(
        "You are a general support assistant.\n"
        "Use the following knowledge to help the customer:\n\n{context}\n\n"
        "Business hours: 9 AM – 6 PM. Cancellations? Ask reason. Discounts? Up to 30% for students.\n"
        "If user says scam/fraud, reply: 'Ok, please contact our service number - 0123456789 - Don't panic.'\n"
        "Don't use ** in responses.\n\nQuery: {query}"
    )
    chain = prompt | llm
    return {"response": chain.invoke({"context": context, "query": state["query"]}).content.strip()}



def generate_response(query: str, context: str, category: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful {category} support agent.\n"
        "Use the following knowledge base info to answer:\n\n{context}\n\n"
        "Query: {query}\n\n"
        "If asked about service hours, please say - Our service hours is from 9 AM to 6 PM, after that you can contact at our email support@supportify.com"
        "If you don't know, say so. Be clear and concise."
    )
    return (prompt | llm).invoke({"context": context, "query": query, "category": category}).content.strip()

def run_customer_support(query: str, kb_chunks: List[Dict]) -> Dict[str, str]:
    category = categorize(query)
    sentiment = analyze_sentiment(query)
    critical_keywords = [
        "urgent", "immediately", "not working", "down", "error", "fail", "scam", "scammed", "cancel my account", "fraud"
    ]
    is_critical = any(kw in query.lower() for kw in critical_keywords)
    if sentiment == "Negative" and is_critical:
        response = escalate(query)
    else:
        relevant_chunks = find_relevant_chunks(query, kb_chunks)
        context = format_context(relevant_chunks) if relevant_chunks else "No relevant info found."
        response = generate_response(query, context, category)
    return {
        "category": category,
        "sentiment": sentiment,
        "response": response
    }

def format_markdown_response(result: Dict[str, str], query: str) -> str:
    lines = [
        "Welcome to the 24/7 Customer Support Agent Platform - Thanks for your query",
        f"Query: {query}",
        f"Category: {result['category']}",
        f"Sentiment: {result['sentiment']}",
        "Response:",
        result["response"],
        "====================================="
    ]
    return "\n".join(lines)

if __name__ == "__main__":
    print("Loading knowledge base...")
    kb_chunks = chunk_with_overlap("knowledge_base.json")
    print(f"Loaded {len(kb_chunks)} chunks. Ready!")
    print("Type 'exit' to quit.\n")
    while True:
        user_query = input("Enter your query: ").strip()
        if user_query.lower() in ("exit", "quit"):
            print("Exiting. Thank you!")
            break
        result = run_customer_support(user_query, kb_chunks)
        print("\n" + format_markdown_response(result, user_query) + "\n")





# from typing import Dict, TypedDict
# from langgraph.graph import StateGraph, END
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from dotenv import load_dotenv
# import json
# import re

# # Load API key
# load_dotenv()
# os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

# # --- Knowledge Base Loading and Chunking ---
# def chunk_with_overlap(kb_path: str):
#     with open(kb_path, 'r', encoding='utf-8') as f:
#         raw_data = json.load(f)
#     splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
#     chunks = []
#     for item in raw_data:
#         split_texts = splitter.split_text(item['text'])
#         for text in split_texts:
#             chunks.append(Document(
#                 page_content=text,
#                 metadata={"folder": item['folder'], "file": item['file']}
#             ))
#     return chunks

# # Build vector index
# kb_docs = chunk_with_overlap("knowledge_base.json")
# embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# vectorstore = FAISS.from_documents(kb_docs, embedding)

# # --- Improved Retrieval ---
# def rephrase_query(query: str) -> str:
#     prompt = ChatPromptTemplate.from_template(
#         "The following user query returned poor results.\n"
#         "Rephrase it into a clearer support-related or technical question:\n\nQuery: {query}"
#     )
#     chain = prompt | llm
#     return chain.invoke({"query": query}).content.strip()

# def retrieve_relevant_chunks(query: str, top_k: int = 8, score_threshold: float = 0.75):
#     results = vectorstore.similarity_search_with_score(query, k=top_k)
#     filtered = [doc for doc, score in results if score >= score_threshold]

#     if len(filtered) < 2:
#         reformulated = rephrase_query(query)
#         results = vectorstore.similarity_search_with_score(reformulated, k=top_k)
#         filtered = [doc for doc, score in results if score >= score_threshold]

#     return filtered

# def format_context(chunks) -> str:
#     context = []
#     for chunk in chunks:
#         meta = chunk.metadata
#         context.append(f"📄 Source: {meta['folder']}/{meta['file']}")
#         context.append(chunk.page_content)
#     return "\n\n".join(context)

# # --- Agent Workflow ---
# class State(TypedDict):
#     query: str
#     category: str
#     sentiment: str
#     response: str

# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# def categorize(state: State) -> State:
#     prompt = ChatPromptTemplate.from_template(
#         "Categorize the following customer query into one of these categories ONLY: Technical, Billing, General.\n"
#         "Return only the category name.\n\nQuery: {query}"
#     )
#     chain = prompt | llm
#     return {"category": chain.invoke({"query": state["query"]}).content.strip()}

# def analyze_sentiment(state: State) -> State:
#     prompt = ChatPromptTemplate.from_template(
#         "Analyze the sentiment of this customer query. Respond with 'Positive', 'Neutral', or 'Negative'.\n\nQuery: {query}"
#     )
#     chain = prompt | llm
#     return {"sentiment": chain.invoke({"query": state["query"]}).content.strip()}

# def handle_technical(state: State) -> State:
#     chunks = retrieve_relevant_chunks(state["query"])
#     context = format_context(chunks)
#     prompt = ChatPromptTemplate.from_template(
#         "You are a helpful and friendly technical support agent.\n"
#         "Use the following knowledge base info to answer:\n\n{context}\n\n"
#         "Query: {query}\n\n"
#         "Don't add ** in responses.\n"
#         "This is a critical technical issue. We'll connect you to a human agent shortly.\n"
#         "Business hours: 9 AM – 6 PM, Mon–Fri. Contact: support@example.com"
#     )
#     chain = prompt | llm
#     return {"response": chain.invoke({"context": context, "query": state["query"]}).content.strip()}

# def handle_billing(state: State) -> State:
#     chunks = retrieve_relevant_chunks(state["query"])
#     context = format_context(chunks)
#     prompt = ChatPromptTemplate.from_template(
#         "You are a polite billing support agent.\n"
#         "Use the following info from our knowledge base:\n\n{context}\n\n"
#         "Answer the billing query clearly:\n\nQuery: {query}\n\n"
#         "Don't use ** in lines.\n"
#         "Handle refunds, duplicate charges, receipts, or payment queries accordingly."
#     )
#     chain = prompt | llm
#     return {"response": chain.invoke({"context": context, "query": state["query"]}).content.strip()}

# def handle_general(state: State) -> State:
#     chunks = retrieve_relevant_chunks(state["query"])
#     context = format_context(chunks)
#     prompt = ChatPromptTemplate.from_template(
#         "You are a general support assistant.\n"
#         "Use the following knowledge to help the customer:\n\n{context}\n\n"
#         "Business hours: 9 AM – 6 PM. Cancellations? Ask reason. Discounts? Up to 30% for students.\n"
#         "If user says scam/fraud, reply: 'Ok, please contact our service number - 0123456789 - Don't panic.'\n"
#         "Don't use ** in responses.\n\nQuery: {query}"
#     )
#     chain = prompt | llm
#     return {"response": chain.invoke({"context": context, "query": state["query"]}).content.strip()}

# def escalate(state: State) -> State:
#     return {
#         "response": (
#             "We're sorry your issue couldn't be resolved automatically.\n"
#             "Be patient. A human agent will contact you soon.\n"
#             "Or call 0123456789.\n"
#             "Business hours: 9 AM to 6 PM, Mon–Fri.\n"
#             "Contact: customersupport@supportai.com"
#         )
#     }

# def route_query(state: State) -> str:
#     critical_keywords = [
#         "urgent", "immediately", "not working", "down", "error", "fail",
#         "scam", "scammed", "cancel my account", "fraud"
#     ]
#     query_lower = state["query"].lower()
#     is_critical = any(kw in query_lower for kw in critical_keywords)
#     if state["sentiment"] == "Negative" and is_critical:
#         return "escalate"
#     elif state["category"] == "Technical":
#         return "handle_technical"
#     elif state["category"] == "Billing":
#         return "handle_billing"
#     else:
#         return "handle_general"

# # --- LangGraph Build ---
# workflow = StateGraph(State)
# workflow.add_node("categorize", categorize)
# workflow.add_node("analyze_sentiment", analyze_sentiment)
# workflow.add_node("handle_technical", handle_technical)
# workflow.add_node("handle_billing", handle_billing)
# workflow.add_node("handle_general", handle_general)
# workflow.add_node("escalate", escalate)

# workflow.add_edge("categorize", "analyze_sentiment")
# workflow.add_conditional_edges("analyze_sentiment", route_query, {
#     "handle_technical": "handle_technical",
#     "handle_billing": "handle_billing",
#     "handle_general": "handle_general",
#     "escalate": "escalate",
# })
# workflow.add_edge("handle_technical", END)
# workflow.add_edge("handle_billing", END)
# workflow.add_edge("handle_general", END)
# workflow.add_edge("escalate", END)
# workflow.set_entry_point("categorize")
# app = workflow.compile()

# # --- Interface ---
# def format_markdown_response(result: Dict[str, str], query: str) -> str:
#     lines = [
#         "Welcome to the 24/7 Customer Support Agent!",
#         f"Query: {query}",
#         f"Category: {result['category']}",
#         f"Sentiment: {result['sentiment']}",
#         "Response:",
#         result["response"],
#         "====================================="
#     ]
#     return "\n".join(lines)

# def run_customer_support(query: str) -> Dict[str, str]:
#     if len(query) < 3 or not re.search(r'\w', query):
#         return {
#             "category": "Unknown",
#             "sentiment": "Neutral",
#             "response": "Sorry, your query seems unclear. Please rephrase or add more details."
#         }
#     results = app.invoke({"query": query})
#     return {
#         "category": results.get("category", ""),
#         "sentiment": results.get("sentiment", ""),
#         "response": results.get("response", "Sorry, I couldn't understand that. Please rephrase or check our FAQs.")
#     }

# def clean_answer_text(answer):
#     # Remove leading bullets (* or -) and extra spaces at line start
#     cleaned_lines = []
#     for line in answer.split('\n'):
#         cleaned_line = re.sub(r'^\s*[\*\-]\s*', '', line)
#         cleaned_lines.append(cleaned_line)
#     return '\n'.join(cleaned_lines)

# if __name__ == "__main__":
#     print("Welcome to the Customer Support Agent! Type 'exit' to quit.\n")
#     while True:
#         user_query = input("Enter your query: ").strip()
#         if user_query.lower() in ("exit", "quit"):
#             print("Exiting. Thank you!")
#             break
#         result = run_customer_support(user_query)
#         print("\n" + format_markdown_response(result, user_query) + "\n")