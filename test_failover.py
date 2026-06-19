import logging
from llm_factory import FailoverLLMWrapper
from langchain_core.messages import AIMessage

# Capture logs
logging.basicConfig(level=logging.INFO)

class MockLLM:
    def __init__(self, name, should_fail=False, fail_message=""):
        self.name = name
        self.should_fail = should_fail
        self.fail_message = fail_message
        
    def invoke(self, prompt, **kwargs):
        if self.should_fail:
            raise Exception(self.fail_message)
        return AIMessage(content=f"Hello from {self.name}!")

def run_test():
    print("=== Test 1: Standard Gemini Execution ===")
    gemini = MockLLM("Gemini")
    groq = MockLLM("Groq")
    llm = FailoverLLMWrapper(gemini, groq)
    
    response1 = llm.invoke("Hi")
    print(f"-> Response: {response1.content}")
    print(f"-> Active Provider: {llm.current_provider}")
    print(f"-> Fallback Count: {llm.fallback_count}")
    
    print("\n=== Test 2: Gemini Hits Rate Limit (429) -> Fails Over to Groq ===")
    gemini_fail = MockLLM("Gemini", should_fail=True, fail_message="429 Resource Exhausted")
    llm2 = FailoverLLMWrapper(gemini_fail, groq)
    
    response2 = llm2.invoke("Hi")
    print(f"-> Response: {response2.content}")
    print(f"-> Active Provider: {llm2.current_provider}")
    print(f"-> Fallback Count: {llm2.fallback_count}")
    
    print("\n=== Test 3: BOTH Gemini AND Groq fail ===")
    groq_fail = MockLLM("Groq", should_fail=True, fail_message="Groq Rate Limit Reached")
    llm3 = FailoverLLMWrapper(gemini_fail, groq_fail)
    
    response3 = llm3.invoke("Hi")
    print(f"-> Response: {response3.content}")
    print(f"-> Active Provider: {llm3.current_provider}")
    print(f"-> Fallback Count: {llm3.fallback_count}")

if __name__ == "__main__":
    run_test()
