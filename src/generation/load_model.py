"""
LoadModel Module.

This module provides a LoadModel class for loading and configuring a Hugging Face model with 4-bit quantization.

Classes:
    LoadModel: A class to log in to the Hugging Face Model Hub, configure 4-bit quantization, set attention implementation,
    load the tokenizer and model, and retrieve GPU memory information.
"""

import logging
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.utils import is_flash_attn_2_available

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LoadModel:
    env_path = Path().cwd() / ".env"

    def __init__(self, model_id: str = "google/gemma-2-2b-it"):
        self.model_id = model_id
        self._logging_to_hugging_face()
        self.nf4_config = self._configure_4bit_quantization()
        self.attn_implementation = self._set_attention_implementation()
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

    def _logging_to_hugging_face(self) -> None:
        """Logs in to the Hugging Face Model Hub using credentials from the .env file.

        Raises:
            ValueError: If the secret token is not found in the .env file.
        """
        load_dotenv(dotenv_path=self.env_path)
        hf_token = os.getenv("HUGGING_FACE_ACCESS_TOKEN")
        if hf_token is None:
            logger.info("Secret token not found in '.env' file !")
            raise ValueError("Secret token not found in '.env' file !")

        login(token=hf_token)
        logger.info("Sucessfully logged to HuggingFace Model Hub.")

    def _configure_4bit_quantization(self) -> BitsAndBytesConfig:
        """Configures 4-bit quantization settings for the model.

        Returns:
            BitsAndBytesConfig: The configuration for 4-bit quantization.
        """
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            if is_flash_attn_2_available()
            else torch.float16,
        )
        logger.info("4-bit quantization configuration set.")
        return nf4_config

    def _set_attention_implementation(self) -> str:
        """Sets the attention implementation based on available hardware capabilities.

        Returns:
            str: The attention implementation to use.
        """
        if (is_flash_attn_2_available()) and (
            torch.cuda.get_device_capability(0)[0] >= 8
        ):
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "sdpa"

        logger.info(f"Attention implementation set to '{attn_implementation}'.")
        return attn_implementation

    def _load_tokenizer(self) -> AutoTokenizer:
        """Loads the tokenizer for the specified model.

        Raises:
            ValueError: If the tokenizer fails to load.

        Returns:
            AutoTokenizer: The tokenizer for the model.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if tokenizer is None:
            logger.error(f"Failed to load tokenizer for model id '{self.model_id}'!")
            raise ValueError(
                f"Failed to load tokenizer for model id '{self.model_id}'!"
            )
        return tokenizer

    def _load_model(self) -> AutoModelForCausalLM:
        """Loads the model with the specified configuration and quantization settings.

        Raises:
            ValueError: If the model fails to load.

        Returns:
            AutoModelForCausalLM: The loaded model.
        """
        try:
            config = AutoConfig.from_pretrained(self.model_id)
            config.hidden_activation = "gelu_pytorch_tanh"

            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.model_id,
                config=config,
                quantization_config=self.nf4_config,
                low_cpu_mem_usage=True,
                attn_implementation=self.attn_implementation,
            )
        except Exception as e:
            logger.error(
                f"Failed to load model for model id '{self.model_id}'! Error: {e}"
            )
            raise ValueError(
                f"Failed to load model for model id '{self.model_id}'! Error: {e}"
            )

        logger.info("Model loaded successfully.")
        return model

    @property
    def gpu_memory_info(self) -> dict:
        """Retrieves information about the GPU memory if CUDA is available.

        Returns:
            dict: A dictionary containing the following keys:
            * total_gpu_memory_gb (float): Total GPU memory in gigabytes.
            * reserved_gpu_memory_gb (float): Reserved GPU memory in gigabytes.
            * allocated_gpu_memory_gb (float): Allocated GPU memory in gigabytes.
            * free_gpu_memory_gb (float): Free GPU memory in gigabytes.
        If CUDA is not available, logs a message and returns an empty dictionary.
        """

        if torch.cuda.is_available():
            total_gpu_memory = torch.cuda.mem_get_info()[1]
            reserved_gpu_memory = torch.cuda.memory_reserved(0)
            allocated_gpu_memory = torch.cuda.memory_allocated(0)
            free_gpu_memory = reserved_gpu_memory - allocated_gpu_memory

            bytes_to_gb = 1024**3
            total_gpu_memory_gb = total_gpu_memory / bytes_to_gb
            reserved_gpu_memory_gb = reserved_gpu_memory / bytes_to_gb
            allocated_gpu_memory_gb = allocated_gpu_memory / bytes_to_gb
            free_gpu_memory_gb = free_gpu_memory / bytes_to_gb

            result = {
                "total_gpu_memory_gb": total_gpu_memory_gb,
                "reserved_gpu_memory_gb": reserved_gpu_memory_gb,
                "allocated_gpu_memory_gb": allocated_gpu_memory_gb,
                "free_gpu_memory_gb": free_gpu_memory_gb,
            }

            return result

        else:
            logger.warning("CUDA is not available, no memory information to get.")
            return {}

    @property
    def get_model_nb_params_and_mem_size(self, model: torch.nn.Module) -> dict:
        """Calculate the number of parameters and memory size of a PyTorch model.

        Args:
            model (torch.nn.Module): The PyTorch model to analyze.
        Returns:
            dict: A dictionary containing:
            * "model_nb_params" (int): The total number of parameters in the model.
            * "model_memory_requirements_gb" (float): The total memory requirements of the model in gigabytes.
        """

        # Calculate the number of parameters and memory size of the model
        nb_parameters = sum([parameter.numel() for parameter in model.parameters()])

        # Calculate the memory requirements of the model
        mem_parameters = sum(
            [
                parameter.nelement() * parameter.element_size()
                for parameter in model.parameters()
            ]
        )
        mem_buffers = sum(
            [buffer.nelement() * buffer.element_size() for buffer in model.buffers()]
        )
        mem_total = mem_parameters + mem_buffers

        result = {
            "model_nb_params": nb_parameters,
            "model_memory_requirements_gb": mem_total / 1024**3,
        }

        return result
