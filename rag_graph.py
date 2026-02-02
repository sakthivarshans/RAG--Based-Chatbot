import json
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from llm import get_llm
from retriever import get_retriever

class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    confidence: float
    used_context: List[str]

def retrieve(state: AgentState):
    question = state["question"]
    print(f"Retrieving for: {question}")
    
    try:
        retriever = get_retriever()
        docs = retriever.invoke(question)
        context_text = [doc.page_content for doc in docs]
        return {"context": context_text}
    except Exception as e:
        print(f"Retrieval error: {e}")
        return {"context": []}

def generate(state: AgentState):
    question = state["question"]
    context = state["context"]
    
    formatted_context = "\n\n".join(context)
    
    system_prompt = """You are a strict Retrieval-Augmented Generation (RAG) assistant.
You must answer the user's question using ONLY the provided context.
If the answer does not exist in the context, say:
'The information is not available in the provided document.'
Do not add any external knowledge.

You MUST return your response in valid JSON format with the following keys:
- "answer": The string answer.
- "confidence": A float between 0 and 1 indicating confidence.
- "used_context": A list of strings containing the specific chunks used to answer.

Context:
{context}
"""
    
    formatted_system_prompt = system_prompt.format(context=formatted_context)
    
    llm = get_llm()
    
    messages = [
        SystemMessage(content=formatted_system_prompt),
        HumanMessage(content=question)
    ]
    
    response = llm.invoke(messages)
    content = response.content.strip()
    
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    
    try:
        parsed = json.loads(content)
        return {
            "answer": parsed.get("answer", "Error parsing response"),
            "confidence": parsed.get("confidence", 0.0),
            "used_context": parsed.get("used_context", [])
        }
    except json.JSONDecodeError:
        return {
            "answer": content, 
            "confidence": 0.0,
            "used_context": []
        }

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
