import logging
from pathlib import Path
from typing import Union

import yaml
from langchain.prompts import PromptTemplate

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AugmentPrompt:
    def __init__(self, query: str, context: str, prompt_template_filepath: Union[str, Path]):
        self.query = query
        self.context = context
        self.prompt_template_filepath = Path(prompt_template_filepath)
        self._load_prompt_template()

    def _load_prompt_template(self) -> PromptTemplate:
        """Loads promt template from a YAML file.

        Raises:
            FileNotFoundError: If the YAML file does not exists.
            ValueError: If the loaded YAML structure is invalid.

        Returns:
            PromptTemplate: _description_
        """
        if not self.prompt_template_filepath.is_file():
            logger.error(f"YAML file not found: '{self.prompt_template_filepath}'.")
            raise FileNotFoundError(f"YAML file does not exist: '{self.prompt_template_filepath}'.")
        
        with open(self.prompt_template_filepath, "r") as f:
            yaml_data = yaml.safe_load(f)

        if (
            not isinstance(yaml_data, dict)
            or "prompt" not in yaml_data
            or "template" not in yaml_data["prompt"]
        ):
            logger.error("Invalid YAML structure. Expected 'prompt' and 'template' keys.")
            raise ValueError(
                "Invalid YAML structure. Expected 'prompt' and 'template' keys."
            )

        yaml_template = yaml_data["prompt"]["template"]
        self.prompt_template = PromptTemplate.from_template(yaml_template)

        logger.info("Prompt template loaded successfully.")

    def create_custom_prompt(self) -> str:
        yaml_variables = {"query": self.query, "context": self.context}
        self.custom_prompt = self.prompt_template.format(**yaml_variables)
        return self.custom_prompt
