import sys
from pathlib import Path

from flask import Flask, jsonify, request

sys.path.append(str(Path(__file__).resolve().parent.parent))

from generation.augment_prompt import AugmentPrompt
from generation.load_model import LoadModel
from generation.text_retriever import TextRetriever

app = Flask(__name__)

# Set path
pages_text_filepath = Path(__file__).cwd() / "datasets/pages-text.json"
embeddings_filepath = Path(__file__).cwd() / "datasets/embeddings-overlap-50.json"
prompt_template_test_filepath = (
    Path(__file__).cwd() / "templates/prompt_template_test.yaml"
)

# Initialize retriever
retriever = TextRetriever(
    embeddings_filepath=embeddings_filepath, pages_text_filepath=pages_text_filepath
)

# Initialize model
model = LoadModel()


@app.route("/api/query", methods=["GET"])
def query_model():
    data = request.json
    query = data.get("query")

    # Retrieve sources and create context
    context_items = retriever.retrieve_relevant_sources(query).get("outputs")
    context = "\n".join([item.get("text") for item in context_items])

    # Create custom prompt
    custom_prompt = AugmentPrompt(
        query=query,
        context=context,
        prompt_template_filepath=prompt_template_test_filepath,
    ).create_custom_prompt()

    # Generate model output
    chat = [{"role": "user", "content": custom_prompt}]
    prompt = model.tokenizer.apply_chat_template(
        conversation=chat,
        tokenize=False,
        add_generation_prompt=True,
    )
    tokenized_prompt = model.tokenizer(prompt, return_tensors="pt").to("cuda")
    generated_output = model.model.generate(
        **tokenized_prompt, max_new_tokens=256, temperature=0.3, do_sample=True
    )
    decoded_output = model.tokenizer.decode(
        generated_output[0], skip_special_tokens=True
    )
    model_response = decoded_output.split("model")[-1].strip()

    return jsonify({"response": model_response})


if __name__ == "__main__":
    app.run(debug=True)
