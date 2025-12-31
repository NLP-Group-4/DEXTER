"""
Gemma LLM Engine using Ollama for better Mac compatibility
"""

import ollama


class GemmaOllamaEngine:
    """
    Gemma engine that uses Ollama backend for better Mac compatibility.
    Ollama handles device management automatically and works well with MPS.
    """

    def __init__(self, data, model_name="gemma3:4b", temperature=0.3, top_n=1, max_new_tokens=256):
        self.model_name = model_name
        self.temperature = temperature
        self.data = data
        self.top_n = top_n
        self.max_new_tokens = max_new_tokens
        
        print(f"Using Ollama backend with model: {model_name}")
        
        # Verify the model is available
        try:
            models_response = ollama.list()
            if 'models' in models_response:
                available_models = [m.get('model', m.get('name', '')) for m in models_response['models']]
                if model_name not in available_models:
                    print(f"Warning: {model_name} not found in Ollama. Available models: {available_models}")
                    print(f"Attempting to pull {model_name}...")
                    ollama.pull(model_name)
                    print(f"Successfully pulled {model_name}")
        except Exception as e:
            print(f"Note: Could not verify model list ({e}), will attempt to use {model_name} anyway")

    def get_gemma_completion(self, system_prompt: str, user_prompt: str):
        """
        Get completion from Gemma model via Ollama.
        
        Args:
            system_prompt: System-level instructions
            user_prompt: User query/prompt
            
        Returns:
            Generated text response
        """
        messages = [
            {
                "role": "user",
                "content": system_prompt,
            },
            {
                "role": "assistant",
                "content": "Yes I will reason and generate the answer",
            },
            {"role": "user", "content": user_prompt},
        ]
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_new_tokens,
                    "top_k": 10,
                    "top_p": 0.95,
                }
            )
            
            # Extract the generated text from the response
            return response['message']['content']
        except Exception as e:
            print(f"Error generating completion: {e}")
            return ""
