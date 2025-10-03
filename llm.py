import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from config import LLM_MODEL_NAME

class LLMManager:
    """Manages the LLM, RAG chain, and response generation."""

    def __init__(self):
        load_dotenv()
        self.llm = ChatGroq(model=LLM_MODEL_NAME, temperature=0.2)
        
    def get_rag_chain(self, retriever):
        """Creates and returns a conversational RAG chain."""
        
        # 1. Contextualize Question Chain
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        # 2. Answering Chain
        qa_system_prompt = (
            "You are an assistant for question-answering tasks for a college campus. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Be concise and helpful. Provide citations by listing the source document names "
            "after your answer, like this: \n\n*Citations: [document1.pdf, document2.pdf]*"
            "\n\nContext:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        # 3. Combine into the final RAG chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        return rag_chain

    def generate_response(self, rag_chain_with_history, chat_history, user_query, session_id):
        """Generates a response using the RAG chain."""
        return rag_chain_with_history.invoke(
            {"input": user_query, "chat_history": chat_history},
            config={"configurable": {"session_id": session_id}}
        )