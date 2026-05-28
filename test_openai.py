# test_openai.py

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os

def test_openai_llm():
    print("\n--- OPENAI LLM ---")
    llm = ChatOpenAI(
        model="gpt-4o-mini",      # cheap model just for testing
        api_key = ("your_Api_key"),
        temperature=0.0,
    )
    response = llm.invoke("Reply with one word: Hello")
    print(f"Response: {response.content}")
    print("✓ OpenAI LLM working")

def test_openai_embeddings():
    print("\n--- OPENAI EMBEDDINGS ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key = ("your_Api_key"),
    )
    vector = embeddings.embed_query("customs duty on electronics")
    print(f"Vector dimensions: {len(vector)}")
    print("✓ OpenAI Embeddings working")

if __name__ == "__main__":
    test_openai_llm()
    print("\n✓ OpenAI tests passed")
