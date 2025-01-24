from pathlib import Path

from augment_prompt import AugmentPrompt
from load_model import LoadModel
from text_retriever import TextRetriever

# Set path
pages_text_filepath = Path(__file__).cwd() / "datasets/pages-text.json"
embeddings_filepath = Path(__file__).cwd() / "datasets/embeddings-overlap-50.json"
prompt_template_test_filepath = (
    Path(__file__).cwd() / "templates/prompt_template_test.yaml"
)


def main(query: str):
    # 1. Retrieve
    retriever = TextRetriever(
        embeddings_filepath=embeddings_filepath, pages_text_filepath=pages_text_filepath
    )
    context_items = retriever.retrieve_relevant_sources(query).get("outputs")
    context = "\n".join([item.get("text") for item in context_items])

    # 2. Augment
    custom_prompt = AugmentPrompt(
        query=query,
        context=context,
        prompt_template_filepath=prompt_template_test_filepath,
    ).create_custom_prompt()

    # 3. Generate
    model = LoadModel()

    # Create prompt template for instruction-tuned model
    chat = [{"role": "user", "content": custom_prompt}]

    # Apply the chat template
    prompt = model.tokenizer.apply_chat_template(
        conversation=chat,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize the input prompt and move it to the GPU
    tokenized_prompt = model.tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate model output
    generated_output = model.model.generate(
        **tokenized_prompt, max_new_tokens=64, temperature=0.2, do_sample=True
    )

    # Decode the model output
    decoded_output = model.tokenizer.decode(
        generated_output[0], skip_special_tokens=True
    )
    model_response = decoded_output.split("model")[-1].strip()

    # Print the model output
    print(f"Model output :\n{model_response}")


if __name__ == "__main__":
    main(query="what is dataplex ?")
