from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config.settings import settings
from typing import List, Dict, Any

class LLMGenerator:
    def __init__(self, model_name: str = settings.LLM_MODEL_NAME, temperature: float = settings.LLM_TEMPERATURE):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=settings.OPENAI_API_KEY)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Answer the user's question based ONLY on the following context:\n\n{context}\n\nIf the answer cannot be found in the context, just say that you don't know, don't try to make up an answer."),
            ("human", "{question}")
        ])

    def generate(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        # Assemble context
        context_text = "\n\n---\n\n".join([doc.get("text", "") for doc in context_docs])
        
        # Create prompt
        messages = self.prompt_template.format_messages(context=context_text, question=query)
        
        # Generate answer
        response = self.llm.invoke(messages)
        return response.content
