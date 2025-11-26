#!/usr/bin/env python3
"""
Interactive CLI Chat AI for code generation using RAG and Persistent Memory.
Now supports both general chat and structured code generation.
"""
import os
import sys
import json
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from langchain_chroma import Chroma
# IMPORT THE NEW FUNCTION:
from tools import generate_tests, save_files, PERSIST_DIR, generate_chat_response 
from ingest import create_embeddings 
from langchain_core.documents import Document 
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

load_dotenv()

# Configuration
K_VALUE = 4 
HISTORY_FILE = os.getenv("HISTORY_FILE", "chat_history.json") 


# --- Helper Functions (Same as before) ---
def serialize_message(message: BaseMessage) -> Dict[str, Any]:
    """Converts a LangChain BaseMessage object to a serializable dictionary."""
    return {
        "type": message.type,
        "content": message.content,
        "example": getattr(message, "example", False), # Ensure 'example' is handled safely
    }

def deserialize_message(data: Dict[str, Any]) -> BaseMessage:
    """Converts a serialized dictionary back to a LangChain BaseMessage object."""
    msg_type = data.get("type", "human")
    content = data.get("content", "")
    
    if msg_type == "system":
        return SystemMessage(content=content)
    elif msg_type == "ai":
        return AIMessage(content=content)
    return HumanMessage(content=content)

def load_history() -> List[BaseMessage]:
    """Loads message history from the history file."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            history = [deserialize_message(d) for d in data]
            print(f"Loaded {len(history)} messages from previous session.")
            return history
    except Exception as e:
        print(f"Warning: Failed to load chat history ({e}). Starting fresh.")
        return []

def save_history(history: List[BaseMessage]):
    """Saves the current message history to the history file."""
    try:
        serializable_history = [serialize_message(msg) for msg in history]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable_history, f, indent=2)
    except Exception as e:
        print(f"Error: Failed to save chat history: {e}")

def initialize_retriever():
    """Initializes and returns the Chroma VectorStoreRetriever."""
    # ... (same as before) ...
    try:
        print("Initializing embeddings and vector database...")
        embeddings = create_embeddings() 
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": K_VALUE})
        print(f"Retriever initialized. Will search for top {K_VALUE} context chunks.")
        return retriever
        
    except Exception as e:
        print(f"Error initializing vector store. Have you run 'python ingest.py' yet?")
        print(f"Details: {e}")
        sys.exit(1)


# --- New Helper Function ---

def is_code_generation_request(request: str) -> bool:
    """Simple check to determine if the request is for code generation."""
    request_lower = request.lower()
    # Define keywords that trigger the structured code generation path
    keywords = ["create", "generate", "write", "code", "script", "test file", "testscript", "function", "class", "module"]
    return any(k in request_lower for k in keywords)


# --- Core Processing Function (Modified) ---

def process_request(user_request: str, retriever, history: List[BaseMessage]) -> Dict[str, Any]:
    """Routes the request to either structured code generation or general chat."""
    
    is_code_request = is_code_generation_request(user_request)

    # 1. Setup System Message based on request type
    if is_code_request:
        # System message for structured output (code generation)
        system_text = (
            "You are an assistant that writes runnable test scripts. Your task is to ONLY return a JSON object that strictly adheres to the provided schema. "
            "Use the CONTEXT EXAMPLES for style and structure. DO NOT respond with conversational text."
        )
    else:
        # System message for general conversation
        system_text = (
            "You are a helpful and friendly code assistant. Answer the user's general questions conversationally, drawing on memory where relevant. "
            "If the user asks you to generate code, politely ask them to include keywords like 'create script' or 'generate test' so you can use your specialized code generation tool."
        )

    # 2. Handle Retrieval (Only relevant for code generation)
    context = []
    if is_code_request:
        print(f"\n🧠 Retrieving context for code request: '{user_request[:50]}...'")
        docs: List[Document] = retriever.invoke(user_request) 
        for d in docs:
            meta = d.metadata
            filename = meta.get("filename", meta.get("source", "unknown"))
            piece = f"File: {filename}\n{d.page_content}"
            context.append(piece)
        print(f"Retrieved {len(context)} examples for style guidance.")

    # 3. Construct Messages List
    new_human_message = HumanMessage(content=user_request)
    messages_for_llm = [SystemMessage(content=system_text)] + history + [new_human_message]

    try:
        if is_code_request:
            # --- CODE GENERATION PATH ---
            print("✨ Generating structured script with LLM...")
            
            # The prompt includes the context for RAG
            context_prompt = "\n\n---\n\n".join(context) if context else "No examples provided."
            human_text_for_tool = (
                f"CONTEXT EXAMPLES (style hints):\n{context_prompt}\n\n"
                f"USER REQUEST:\n{user_request}\n\n"
                "Return only a JSON object that matches the schema. Generate 1 to 3 tests as needed."
            )
            
            # Note: We still pass the full messages_for_llm to `generate_tests` if that function is updated to use it
            # But the original generate_tests only accepts prompt_request, so we pass the context-rich human_text_for_tool
            tests = generate_tests(human_text_for_tool, context) 
            
            # Update history: append original request and AI summary
            ai_response_summary = f"Generated {len(tests)} script(s): " + ", ".join([t.filename for t in tests])
            history.append(new_human_message)
            history.append(AIMessage(content=ai_response_summary))
            
            return {"type": "code", "tests": tests}
        
        else:
            # --- CHAT / CONVERSATION PATH ---
            print("💬 Generating conversational response...")
            
            chat_response = generate_chat_response(messages_for_llm)
            
            # Update history: append original request and AI response
            history.append(new_human_message)
            history.append(AIMessage(content=chat_response))
            
            return {"type": "chat", "response": chat_response}
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return {"type": "error"}


def cli_loop():
    """The main interactive chat loop."""
    print("--- CLI Code Generator AI Initialized ---")
    retriever = initialize_retriever()
    
    conversation_history = load_history()
    print(f"Total messages in history: {len(conversation_history)}")
    
    print("Type your request. Use keywords like 'create script' or 'generate test' for code generation.")
    print("----------------------------------------------------------------------------------------\n")
    
    while True:
        try:
            user_input = input("Request > ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit"]:
                print("\nSaving conversation history...")
                save_history(conversation_history) 
                print("Goodbye!")
                break
            
            result = process_request(user_input, retriever, conversation_history)
            
            if result["type"] == "error":
                print("Processing failed. Please check logs and try again.")
                continue

            # Print results based on type
            print("\n--- Model Output ---")
            if result["type"] == "chat":
                # Print conversational response
                print(result["response"])
                
            elif result["type"] == "code":
                # Print generated code
                generated_tests = result["tests"]
                if not generated_tests:
                    print("Code generation failed or returned no scripts.")
                else:
                    for t in generated_tests:
                        print(f"**Filename:** {t.filename}")
                        print(f"**Description:** {t.description}")
                        print("\n```python")
                        print(t.content)
                        print("```\n")
                        
                    # Offer to save the files (only for code)
                    save_choice = input("Save generated files? (y/n) [n]: ").lower()
                    if save_choice == 'y':
                        save_result = save_files(generated_tests)
                        if save_result.get('ok'):
                            print(f"✅ Successfully saved files to ./generated_tests/")
                        else:
                            print(f"❌ Error saving files: {save_result.get('files')}")

            print("--------------------\n")

        except KeyboardInterrupt:
            print("\nSaving conversation history...")
            save_history(conversation_history) 
            print("Goodbye!")
            break
        except EOFError:
            print("\nSaving conversation history...")
            save_history(conversation_history) 
            print("Goodbye!")
            break


if __name__ == "__main__":
    cli_loop()