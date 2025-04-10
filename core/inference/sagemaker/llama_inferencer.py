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
    
    def generate_text(self, user_query: str, default_prompt: str, context: List[Dict] = None, **kwargs) -> str:
        """
        Generates a response based on the provided user query and context. It formats the context, sends it to 
        the model for text generation, and processes the response to return the generated text.

        Args:
            user_query (str): The query provided by the user for which a response is generated.
            context (List[Dict]): A list of context passages, each represented as a dictionary.
            default_prompt (str): A default prompt that is used to guide the text generation.
            **kwargs: Additional keyword arguments, if any.

        Returns:
            tuple: A tuple containing metadata (str) and the cleaned generated text (str).
        """
        
        # Ensure the generation predictor is initialized
        if not self.inferencing_predictor:
            raise ValueError("Generation predictor not initialized")
        
        system_prompt, prompt = self.generate_prompt(self.experiment_config, default_prompt, user_query, context)
        
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

        try:
            start_time = time.time()
            
            # Get response from the model
            response = self.inferencing_predictor.predict(payload)

            # Calculate latency metrics
            latency = int((time.time() - start_time) * 1000)
            generated_text = response["choices"][0]["message"]["content"]
            
            # Process the generated text to extract the answer
            if "The final answer is:" in generated_text:
                answer = generated_text.split("The final answer is:")[1].strip()
            elif "Assistant:" in generated_text:
                answer = generated_text.split("Assistant:")[1].strip()
            else:
                answer = generated_text.strip()

            # Clean and validate the response
            cleaned_response = self._clean_response(answer)
            
            # Final validation of the generated text
            if not cleaned_response or cleaned_response.isspace() or 'DRAFT' in cleaned_response:
                return "Unable to generate a proper response. Please try again."

            # SageMaker does not provide input tokens as metadata.
            # As a workaround, we use a rough approximation: ~4 characters per token.
            input_tokens = len(prompt) // 4
            output_tokens = len(generated_text) // 4
            total_tokens = input_tokens + output_tokens
            
            answer_metadata = {
                'inputTokens': input_tokens,
                'outputTokens': output_tokens,
                'totalTokens': total_tokens,
                'latencyMs': latency
            }
            
            return answer_metadata, cleaned_response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
        