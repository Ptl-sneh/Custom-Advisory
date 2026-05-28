# test_setup.py

import logging
logging.basicConfig(level=logging.INFO)

def test_config():
    print("\n--- CONFIG ---")
    from config import settings
    print(f"LLM:        {settings.LLM_PROVIDER} / {settings.LLM_MODEL}")
    print(f"Embeddings: {settings.EMBEDDING_PROVIDER} / {settings.EMBEDDING_MODEL}")
    print(f"Qdrant:     {settings.QDRANT_MODE} → collection: {settings.QDRANT_COLLECTION_NAME}")
    print(f"Langfuse:   {settings.IS_LANGFUSE_ENABLED}")
    print("✓ Config loaded")

def test_llm():
    print("\n--- LLM ---")
    from providers.llm_factory import get_llm
    llm = get_llm()
    response = llm.invoke("Reply with one word: Hello")
    print(f"Response: {response.content}")
    print("✓ LLM working")

def test_embeddings():
    print("\n--- EMBEDDINGS ---")
    from providers.llm_factory import get_embeddings
    embeddings = get_embeddings()
    vector = embeddings.embed_query("customs duty on electronics")
    print(f"Vector dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
    print("✓ Embeddings working")

def test_qdrant():
    print("\n--- QDRANT ---")
    from providers.llm_factory import get_qdrant_client
    client = get_qdrant_client()
    collections = client.get_collections()
    print(f"Existing collections: {[c.name for c in collections.collections]}")
    print("✓ Qdrant connected")

if __name__ == "__main__":
    test_config()
    test_llm()
    test_embeddings()
    test_qdrant()
    print("\n✓ All checks passed")