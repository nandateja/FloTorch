from typing import List, Dict
import logging
import time
import random
from config.experimental_config import ExperimentalConfig
from .sagemaker_inferencer import SageMakerInferencer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class LlamaInferencer(SageMakerInferencer):
    def __init__(self, model_id: str, experiment_config: ExperimentalConfig, region: str, role_arn: str):
        super().__init__(model_id, experiment_config, region, role_arn)
        
    def _prepare_conversation(self, message: str, role: str):
        # Format message and role into a conversation
        if not message or not role:
            logger.error(f"Error in parsing message or role")
        conversation = {
                "role": role, 
                "content": message
            }
        return conversation
    
    def generate_prompt(self, experiment_config: ExperimentalConfig, default_prompt: str, user_query: str, context: List[Dict] = None):
        n_shot_prompt_guide = experiment_config.n_shot_prompt_guide_obj
        n_shot_prompt = experiment_config.n_shot_prompts
        # Input validation
        if n_shot_prompt < 0:
            raise ValueError("n_shot_prompt must be non-negative")
        
        # Get system prompt
        system_prompt = default_prompt if n_shot_prompt_guide is None or n_shot_prompt_guide.system_prompt is None else n_shot_prompt_guide.system_prompt
        
        context_text = ""
        if context:
            context_text = self._format_context(user_query, context)
        
        base_prompt = n_shot_prompt_guide.user_prompt if n_shot_prompt_guide.user_prompt else ""
        
        if n_shot_prompt == 0:
            logger.info("into zero shot prompt")
    
            messages = []
            messages.append(self._prepare_conversation(role="user", message=base_prompt))
            if context_text:
                messages.append(self._prepare_conversation(role="user", message=context_text))
            messages.append(self._prepare_conversation(role="user", message=user_query))

            return system_prompt, messages

        # Get examples if nshot is not zero
        examples = n_shot_prompt_guide.examples
        
        # Format examples
        selected_examples = (random.sample(examples, n_shot_prompt) 
                        if len(examples) > n_shot_prompt 
                        else examples)
        
        logger.info(f"into {n_shot_prompt} shot prompt  with examples {len(selected_examples)}")
        
        messages = []
        messages.append(self._prepare_conversation(role="user", message=base_prompt))
        for example in selected_examples:
            if 'example' in example:
                messages.append(self._prepare_conversation(role="user", message=example['example']))
            elif 'question' in example and 'answer' in example:
                messages.append(self._prepare_conversation(role="user", message=example['question']))
                messages.append(self._prepare_conversation(role="assistant", message=example['answer']))
        
        if context_text:
            messages.append(self._prepare_conversation(role="user", message=context_text))
            
        messages.append(self._prepare_conversation(role="user", message=user_query))

        return system_prompt, messages
        
    def construct_payload(self, system_prompt: str, prompt: str) -> dict:
        """
        Constructs llama 4 payload dictionary for model inference with the given prompts and default parameters.
        
        Args:
            system_prompt (str): The system-level prompt that guides the model's behavior
            prompt (str): The actual prompt/query to be sent to the model

        """
        # Define default parameters for the model's generation
        default_params = {
            "max_new_tokens": 256,
            "temperature": self.experiment_config.temp_retrieval_llm,
            "top_p": 0.9,
            "do_sample": True
        }
        
        # Prepare payload for model inference
        payload = {
            "system": system_prompt,
            "messages": prompt,
            "parameters": default_params
            }
        
        return payload
    
    def parse_response(self, response: dict) -> str:
        """
        Parses the response from the model and extracts the generated text.

        Args:
            response (dict): The raw response from the model
        """
        if "choices" in response and isinstance(response["choices"], list):
            return response["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"Unexpected Llama-4 response format: {response}")
        