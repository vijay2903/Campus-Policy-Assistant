# This script is for offline experimentation to find the best strategies.
# It requires manual setup and is not part of the main Streamlit app.

import time
from vectorstore import VectorStoreManager
from llm import LLMManager

def run_evaluation():
    print("Initializing components for evaluation...")
    vsm = VectorStoreManager()
    llm = LLMManager()

    # --- Define your evaluation set ---
    test_queries = [
        "What is the policy on hostel room changes?",
        "How do I file a complaint for a broken fan in my room?",
        "What are the library hours during examination week?",
        "Summarize the campus code of conduct regarding academic integrity."
    ]

    chunking_strategies = ["recursive", "fixed_size", "semantic"]
    search_strategies = ["similarity", "mmr", "hybrid"]

    results = {}

    print("Starting evaluation...")
    for chunk_strat in chunking_strategies:
        for search_strat in search_strategies:
            strategy_key = f"Chunk: {chunk_strat} | Search: {search_strat}"
            print(f"\n--- Testing Strategy: {strategy_key} ---")
            results[strategy_key] = []
            
            # Note: For a true evaluation, the vector store should be rebuilt
            # with the specific chunking strategy. This script simplifies by
            # using the default chunking for the pre-built admin store.
            # A more advanced script would rebuild the store for each strategy.
            retriever = vsm.get_retriever(user_vs=None, search_type=search_strat)
            rag_chain = llm.get_rag_chain(retriever)

            for query in test_queries:
                print(f"Querying: {query}")
                start_time = time.time()
                
                response = rag_chain.invoke({"input": query, "chat_history": []})
                
                end_time = time.time()
                latency = end_time - start_time
                
                answer = response.get("answer", "No answer found.")
                context_docs = response.get("context", [])
                
                # Basic evaluation metrics
                eval_metrics = {
                    "query": query,
                    "answer": answer,
                    "retrieved_docs": len(context_docs),
                    "latency_seconds": round(latency, 2),
                    # You could add more advanced metrics here (e.g., LLM-as-judge)
                }
                results[strategy_key].append(eval_metrics)
                print(f"Latency: {latency:.2f}s, Retrieved docs: {len(context_docs)}")

    # --- Print Results ---
    print("\n\n--- EVALUATION COMPLETE ---")
    for strategy, evals in results.items():
        print(f"\n--- Strategy: {strategy} ---")
        total_latency = sum(e['latency_seconds'] for e in evals)
        avg_latency = total_latency / len(evals) if evals else 0
        print(f"Average Latency: {avg_latency:.2f}s")
        for e in evals:
            print(f"  Query: {e['query']}")
            print(f"  Answer: {e['answer'][:100]}...") # Print snippet of answer
            print("-" * 20)

if __name__ == "__main__":
    run_evaluation()