from groq import Groq
from typing import List, Dict
import os
from dotenv import load_dotenv

class Generator:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        load_dotenv()
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model_name
        
    def generate(self, question: str, contexts: List[Dict]) -> str:
        """Generate an answer using retrieved passages with Groq's Llama-2."""
        # Combine contexts
        context_texts = [c["text"] for c in contexts]
        combined_context = " ".join(context_texts)
        
        # Create prompt
        prompt = f"""You are a helpful AI assistant. Answer the question based only on the provided context. If the context doesn't contain enough information to answer the question, say so.

Context: {combined_context}

Question: {question}

Answer: Let me answer based on the provided context."""

        # Generate completion
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
            top_p=0.9
        )
        
        return completion.choices[0].message.content 