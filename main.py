from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

texts = [
    "LangChain is used to build LLM-powered applications.",
    "RAG helps LLMs answer questions using external documents.",
    "FAISS is a vector database for similarity search."
]

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

vectorstore = FAISS.from_texts(texts, embeddings)
retriever = vectorstore.as_retriever()

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.
                                          
Context:
{context}

Question:                                                                                              
{question}
""")

rag_chain = (
    {"context":retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

response = rag_chain.invoke("What is Langchain")
print(response.content)