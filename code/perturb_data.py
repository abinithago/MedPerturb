import os
import json
import random
import torch
import argparse
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Literal, Dict, List, Optional
from utils import (
    load_hf_token,
    load_openai_token,
    validate_dataset_type,
    validate_perturbation_type,
    validate_gender_variant,
    validate_viewpoint_variant,
    setup_logging
)

# Define dataset and perturbation types
DatasetType = Literal["oncqa", "askadocs", "usmle"]
PerturbationType = Literal["gender", "stylistic", "viewpoint"]

class ClinicalContextPerturber:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.logger = setup_logging()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
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
        
        # Load OpenAI token
        openai_token = load_openai_token()
        if not openai_token:
            self.logger.warning("No OpenAI token found. Viewpoint perturbations will not work.")
        else:
            openai.api_key = openai_token
        
        self.temperature = 0.5
        self.random_seed = 42
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        self.logger.info(f"Initialized ClinicalContextPerturber with model: {model_name}")

    def _call_gpt4(self, messages: List[Dict[str, str]]) -> str:
        """
        Call GPT-4 API with the given messages.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'
            
        Returns:
            str: The model's response
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=self.temperature,
                max_tokens=512
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error calling GPT-4 API: {str(e)}")
            raise

    def perturb_gender(self, text: str, variant: Literal["swap", "remove"]) -> str:
        """
        Perturb the gender in the given text using either gender-swapping or gender-removal.
        
        Args:
            text (str): The input clinical context
            variant (str): Either "swap" or "remove"
            
        Returns:
            str: The perturbed text
        """
        if not validate_gender_variant(variant):
            raise ValueError(f"Invalid gender variant: {variant}")
            
        if variant == "swap":
            prompt = "Please swap the gender in the following text, making no other changes:\n"
        else:  # remove
            prompt = "Please remove the gender markers in the following text, making no other changes:\n"
            
        full_prompt = f"{prompt}{text}"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the perturbed text from the response
        perturbed_text = response[len(full_prompt):].strip()
        return perturbed_text

    def perturb_stylistic(self, text: str, variant: Literal["uncertain", "colorful"]) -> str:
        """
        Perturb the style of the given clinical context using either uncertain or colorful language.
        
        Args:
            text (str): The input clinical context
            variant (str): Either "uncertain" or "colorful"
            
        Returns:
            str: The perturbed text
        """
        if variant == "uncertain":
            prompt = (
                "Add uncertain language to this patient message to make the patient sound more unsure. "
                "Use hedge words like 'well, kind of, sort of, possibly, maybe' and uncertainty verbs "
                "like 'think, suppose, seem, imagine':\n"
            )
        else:  # colorful
            prompt = (
                "Add colorful language to this patient message to make the patient sound more expressive. "
                "Use exclamations like 'hey, oh,' expletives like 'wow, woah,' and intensive adverbs "
                "like 'really, very, quite':\n"
            )
            
        full_prompt = f"{prompt}{text}"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the perturbed text from the response
        perturbed_text = response[len(full_prompt):].strip()
        return perturbed_text

    def perturb_viewpoint(self, text: str, variant: Literal["multiturn", "summarized"]) -> str:
        """
        Perturb the viewpoint of the given clinical context using either multiturn conversation
        or summarized format using GPT-4.
        
        Args:
            text (str): The input clinical context
            variant (str): Either "multiturn" or "summarized"
            
        Returns:
            str: The perturbed text
        """
        if not validate_viewpoint_variant(variant):
            raise ValueError(f"Invalid viewpoint variant: {variant}")
            
        if variant == "multiturn":
            # Doctor system message
            doctor_system = (
                "You are an AI doctor. Your task is to arrive at a diagnosis of a patient's medical "
                "condition by asking one question at a time. Each question must be no longer than "
                "one line. Continue asking different, non-repetitive questions until you are completely "
                "confident in your diagnosis. Your questions should gather information about the patient's "
                "age, sex, current symptoms, medical history, medications, and relevant family history "
                "if necessary. Keep your questions brief and easy to understand. Once you are certain, "
                "provide a final diagnosis in a short response, stating only the diagnosis name and only "
                "providing a single diagnosis."
            )
            
            # Patient system message
            patient_system = (
                "You are a patient with no medical training. Your job is to respond to the doctor's "
                "questions using only the information provided in the case vignette. You must not reveal "
                "that you are referencing a vignette and continue to speak in first person throughout. "
                "Do not suppose any new symptoms or provide knowledge beyond what is given. Only answer "
                "the specific question asked and keep your response to a single sentence. Use layperson-friendly "
                "language, simplifying any complex terms from the vignette. Your replies should remain "
                "grounded in the provided information."
            )
            
            # Initialize conversation
            doctor_messages = [
                {"role": "system", "content": doctor_system},
                {"role": "user", "content": f"Patient information:\n{text}\n\nStart the conversation with your first question:"}
            ]
            
            patient_messages = [
                {"role": "system", "content": patient_system},
                {"role": "user", "content": f"Your information:\n{text}\n\nRespond to the doctor's questions:"}
            ]
            
            # Generate doctor's first question
            doctor_question = self._call_gpt4(doctor_messages)
            conversation = f"Doctor: {doctor_question}\n"
            
            # Add doctor's question to patient's context
            patient_messages.append({"role": "user", "content": f"Doctor: {doctor_question}"})
            
            # Generate patient's response
            patient_response = self._call_gpt4(patient_messages)
            conversation += f"Patient: {patient_response}\n"
            
            # Continue conversation until diagnosis
            while True:
                # Add patient's response to doctor's context
                doctor_messages.append({"role": "user", "content": f"Patient: {patient_response}"})
                
                # Generate next doctor question
                next_question = self._call_gpt4(doctor_messages)
                
                # Check if this is a diagnosis
                if "diagnosis" in next_question.lower() or "diagnosed" in next_question.lower():
                    conversation += f"Doctor: {next_question}\n"
                    break
                
                conversation += f"Doctor: {next_question}\n"
                
                # Add doctor's question to patient's context
                patient_messages.append({"role": "user", "content": f"Doctor: {next_question}"})
                
                # Generate patient's response
                next_response = self._call_gpt4(patient_messages)
                conversation += f"Patient: {next_response}\n"
                
                # Update patient response for next iteration
                patient_response = next_response
            
            return conversation
            
        else:  # summarized
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are tasked with converting a Query Vignette from first-person to third-person perspective. "
                        "It is essential that you make no changes to the content or add any new information; doing so "
                        "will result in penalties. A demonstrative Example follows the vignette to illustrate the "
                        "expected transformation."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Query Vignette:\n"
                        f"{text}\n\n"
                        "Convert this to third-person perspective, maintaining all the original information but "
                        "changing the perspective:"
                    )
                }
            ]
            
            return self._call_gpt4(messages)

    def perturb_context(
        self,
        text: str,
        dataset_type: DatasetType,
        perturbation_type: PerturbationType,
        **kwargs
    ) -> str:
        """
        Main function to perturb clinical contexts based on the specified type.
        
        Args:
            text (str): The input clinical context
            dataset_type (str): One of "oncqa", "askadocs", or "usmle"
            perturbation_type (str): One of "gender", "stylistic", or "viewpoint"
            **kwargs: Additional arguments for specific perturbation types
            
        Returns:
            str: The perturbed text
        """
        if not validate_dataset_type(dataset_type):
            raise ValueError(f"Invalid dataset type: {dataset_type}")
        if not validate_perturbation_type(perturbation_type):
            raise ValueError(f"Invalid perturbation type: {perturbation_type}")
            
        self.logger.info(f"Perturbing text using {perturbation_type} perturbation for {dataset_type} dataset")
        
        if perturbation_type == "gender":
            variant = kwargs.get("variant", "swap")
            return self.perturb_gender(text, variant)
        elif perturbation_type == "stylistic":
            variant = kwargs.get("variant", "uncertain")
            return self.perturb_stylistic(text, variant)
        elif perturbation_type == "viewpoint":
            variant = kwargs.get("variant", "multiturn")
            return self.perturb_viewpoint(text, variant)
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")

def parse_args():
    parser = argparse.ArgumentParser(description="Clinical Context Perturbation Tool")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="The clinical context text to perturb"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["oncqa", "askadocs", "usmle"],
        help="The dataset type"
    )
    parser.add_argument(
        "--perturbation",
        type=str,
        required=True,
        choices=["gender", "stylistic", "viewpoint"],
        help="The type of perturbation to apply"
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["swap", "remove", "uncertain", "colorful", "multiturn", "summarized"],
        help="Variant for perturbation (gender: swap/remove, stylistic: uncertain/colorful, viewpoint: multiturn/summarized)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="The model to use for perturbation"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path to save the perturbed text (optional)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logging()
    
    try:
        perturber = ClinicalContextPerturber(model_name=args.model)
        
        # Prepare kwargs for perturbation
        kwargs = {}
        if args.perturbation == "gender":
            if not args.variant:
                logger.warning("No variant specified for gender perturbation, defaulting to 'swap'")
                kwargs["variant"] = "swap"
            else:
                kwargs["variant"] = args.variant
        elif args.perturbation == "stylistic":
            if not args.variant:
                logger.warning("No variant specified for stylistic perturbation, defaulting to 'uncertain'")
                kwargs["variant"] = "uncertain"
            else:
                kwargs["variant"] = args.variant
        elif args.perturbation == "viewpoint":
            if not args.variant:
                logger.warning("No variant specified for viewpoint perturbation, defaulting to 'multiturn'")
                kwargs["variant"] = "multiturn"
            else:
                kwargs["variant"] = args.variant
        
        # Perform perturbation
        perturbed_text = perturber.perturb_context(
            text=args.text,
            dataset_type=args.dataset,
            perturbation_type=args.perturbation,
            **kwargs
        )
        
        # Output results
        print(f"Original: {args.text}")
        print(f"Perturbed: {perturbed_text}")
        
        # Save to file if output path is provided
        if args.output:
            with open(args.output, "w") as f:
                json.dump({
                    "original": args.text,
                    "perturbed": perturbed_text,
                    "dataset": args.dataset,
                    "perturbation_type": args.perturbation,
                    "variant": kwargs.get("variant")
                }, f, indent=2)
            logger.info(f"Results saved to {args.output}")
            
    except Exception as e:
        logger.error(f"Error during perturbation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
