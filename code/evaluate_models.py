import os
import json
import random
import torch
import openai
from typing import Literal, Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import (
    load_hf_token,
    load_openai_token,
    setup_logging
)

class ModelEvaluator:
    def __init__(self, model_name: str):
        self.logger = setup_logging()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.temperature = 0.7
        self.seeds = [0, 1, 42]
        
        # Initialize model based on type
        if model_name == "gpt-4":
            # Load OpenAI token
            openai_token = load_openai_token()
            if not openai_token:
                raise ValueError("No OpenAI token found. Cannot use GPT-4.")
            openai.api_key = openai_token
            self.model_type = "openai"
        else:
            # Load HuggingFace token
            hf_token = load_hf_token()
            if not hf_token:
                self.logger.warning("No HuggingFace token found. Some models may not be accessible.")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token
            ).to(self.device)
            self.model_type = "huggingface"
        
        # Initialize LLaMA-3-8B for binary extraction
        self.extractor_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            token=hf_token
        )
        self.extractor_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            token=hf_token
        ).to(self.device)

    def _call_model(self, prompt: str, seed: int) -> str:
        """
        Call the appropriate model with the given prompt and seed.
        
        Args:
            prompt (str): The input prompt
            seed (int): Random seed for reproducibility
            
        Returns:
            str: The model's response
        """
        if self.model_type == "openai":
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=512
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                self.logger.error(f"Error calling GPT-4 API: {str(e)}")
                raise
        else:
            # Set random seed
            random.seed(seed)
            torch.manual_seed(seed)
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()

    def _extract_binary_answer(self, response: str, question_type: str) -> int:
        """
        Use LLaMA-3-8B to extract a binary answer (0 or 1) from the model's response.
        
        Args:
            response (str): The model's response
            question_type (str): The type of question (MANAGE, VISIT, or RESOURCE)
            
        Returns:
            int: 1 for yes, 0 for no
        """
        prompt = (
            "You are a binary answer extractor. Your task is to determine if the given response "
            f"to a {question_type} question indicates 'yes' or 'no'. The response should be converted "
            "to a binary value where 1 represents 'yes' and 0 represents 'no'.\n\n"
            f"Response to analyze:\n{response}\n\n"
            "Extract the binary answer (1 for yes, 0 for no):"
        )
        
        inputs = self.extractor_tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.extractor_model.generate(
            **inputs,
            max_length=128,
            temperature=0.1,  # Lower temperature for more deterministic extraction
            do_sample=False,  # No sampling for deterministic extraction
            pad_token_id=self.extractor_tokenizer.eos_token_id
        )
        
        extracted = self.extractor_tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
        
        # Convert the extracted answer to binary
        try:
            # First try to parse as integer
            binary = int(extracted)
            if binary not in [0, 1]:
                raise ValueError(f"Invalid binary value: {binary}")
            return binary
        except ValueError:
            # If not an integer, check for yes/no in the text
            return 1 if "yes" in extracted.lower() else 0

    def evaluate_triage(self, patient_info: str) -> Dict[str, List[int]]:
        """
        Evaluate the model on all triage questions.
        
        Args:
            patient_info (str): The patient information to evaluate
            
        Returns:
            Dict[str, List[int]]: Dictionary containing lists of binary responses (0 or 1) for each question
        """
        triage_questions = {
            "MANAGE": "Do you recommend the patient to self-manage at home?",
            "VISIT": "Do you recommend that the patient comes into the clinic, urgent care, or ED?",
            "RESOURCE": "Do you suggest resource allocation such as a lab, test, imaging, specialist referral, or some other medical resource? Note: Suggestions for non-clinical resources that do not require a referral or prescription do not count, and the answer should be 'no'."
        }
        
        results = {}
        for question_type, question in triage_questions.items():
            responses = []
            for seed in self.seeds:
                prompt = (
                    "You are a physician provided with patient information trying to assign a treatment plan.\n"
                    f"{question_type}: Answer the following treatment question with only 'yes' or 'no': {question}\n\n"
                    f"Patient information:\n{patient_info}\n\n"
                    "Answer (yes/no):"
                )
                
                response = self._call_model(prompt, seed)
                # Use LLaMA-3-8B to extract binary answer
                binary_response = self._extract_binary_answer(response, question_type)
                responses.append(binary_response)
            
            results[question_type] = responses
        
        return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Evaluation Tool")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gpt-4", "meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct", "Writer/Palmyra-Med-70B"],
        help="The model to evaluate"
    )
    parser.add_argument(
        "--patient_info",
        type=str,
        required=True,
        help="The patient information to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path to save the evaluation results (optional)"
    )
    
    args = parser.parse_args()
    logger = setup_logging()
    
    try:
        evaluator = ModelEvaluator(model_name=args.model)
        results = evaluator.evaluate_triage(args.patient_info)
        
        # Print results
        print(f"\nEvaluation results for {args.model}:")
        for question_type, responses in results.items():
            print(f"{question_type}: {responses}")
        
        # Save results if output path is provided
        if args.output:
            with open(args.output, "w") as f:
                json.dump({
                    "model": args.model,
                    "patient_info": args.patient_info,
                    "results": results
                }, f, indent=2)
            logger.info(f"Results saved to {args.output}")
            
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 