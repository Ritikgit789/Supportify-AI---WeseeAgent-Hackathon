# Supportify AI
Supportify AI is an AI-powered, 24/7 customer support agent designed to provide fast and accurate assistance across multiple customer query categories. It leverages state-of-the-art generative AI (Google Gemini) along with a Retrieval-Augmented Generation (RAG) approach to deliver precise, context-aware responses by consulting a structured knowledge base.
The app is built with a focus on ease of use and seamless integration, deployed as a Streamlit-based web application.

Try Link: https://supportifyweb.streamlit.app/

Features
- AI-powered 24/7 Customer Support Agent: Responds instantly to customer queries at any time.
- Categorizes queries into Technical, Billing, or General categories.
- Sentiment analysis of customer queries (Positive, Neutral, Negative).
- Retrieval-Augmented Generation (RAG): Uses a chunked knowledge base from documents zipped in JSON, to provide contextually relevant answers.
- Fast keyword-based chunk retrieval for relevant knowledge base information.
- Escalation handling for urgent or critical negative queries with human agent contact information.
- Polite fallback responses for out-of-knowledge-base queries.
- Streamlit web UI for easy user interaction.
- Modular and extensible Python codebase for easy maintenance and updates.

Architecture Overview

Knowledge Base Loading
Extracts and loads JSON knowledge base data stored inside a ZIP file.

Chunking
Text data is split into overlapping chunks for better retrieval granularity.

Query Processing

Categorize query into Technical, Billing, or General.

Analyze sentiment (Positive, Neutral, Negative).

Detect critical issues via keywords and sentiment.

Retrieval
Fast keyword intersection-based retrieval selects the top relevant chunks.

Response Generation
Uses Google Gemini LLM prompts with the retrieved context for an accurate reply.

Escalation
Automatically escalates critical negative queries to human agents.


Code Structure

load_kb_from_zip(): Extracts and loads knowledge base JSON from a ZIP archive.
chunk_with_overlap(): Splits knowledge base documents into overlapping chunks for retrieval.
find_relevant_chunks(): Fast keyword search to find relevant chunks for a given query.
categorize(): Uses LLM to categorize queries into Technical, Billing, or General.
analyze_sentiment(): Uses LLM to analyze sentiment of queries.
generate_response(): Generates answers using LLM and relevant context.
escalate(): Provides escalation message for urgent negative queries.
run_customer_support(): Main function orchestrating query handling workflow.
format_markdown_response(): Prepares a formatted string response to display.


How it works

The user submits a query via the Streamlit interface.
The system categorizes the query and analyzes its sentiment.
If the query is critical and negative, an escalation message is shown.
Otherwise, the system finds relevant knowledge base chunks.
If relevant info is found, the AI generates a detailed answer using the context.
If no relevant data is found, the AI provides a general knowledge answer.
The formatted response is displayed back to the user.n function orchestrating query handling workflow.
format_markdown_response(): Prepares a formatted string response to display

Future Updates

Vector-based semantic search integration (FAISS or Pinecone) for better retrieval beyond keyword matching.
Multi-language support to assist customers globally.
Interactive conversation flow with follow-up questions and clarification.
Admin dashboard to update knowledge base and monitor queries.
Integration with CRM and ticketing systems for automatic ticket creation.
Sentiment-driven personalized responses adapting tone and content dynamically.
Expanded escalation workflows with SMS and chat notifications.
Analytics and reporting on common queries, resolution time, and customer satisfaction.
Voice assistant support using speech-to-text and text-to-speech APIs.

Contribution
Contributions are welcome! Please open an issue or submit a pull request with improvements.

Check it out: https://supportifyweb.streamlit.app/

This project is built for upcoming WeSee Agent Hackathon - Japan
