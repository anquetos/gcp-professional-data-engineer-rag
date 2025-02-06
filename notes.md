# Notes

## `curl` query

```
curl -X POST http://127.0.0.1:5000/api/query \
     -H "Content-Type: application/json" \
     -d '{"query": "A team of data warehouse developers is migrating a set of legacy Python scripts that have been used to transform data as part of an ETL process. They would like to use a service that allows them to use Python and requires minimal administration and operations support. Which GCP service would you recommend? A. Cloud Dataproc, B. Cloud Dataflow, C. Cloud Spanner or D. Cloud Dataprep"}'
```



## RAG Project Organization

### Project Root Directory

- `README.md`: Project description, installation instructions, etc.
- `requirements.txt` or `Pipfile`: Python package dependencies.
- `Dockerfile`: Docker image configuration.
- `docker-compose.yml`: For managing Docker services (if needed).

### Directories

- `src/`: Your source code.
  - `__init__.py`: Initializes the package.
  - `preprocessing/`: Text extraction and formatting.
    - `__init__.py`
    - `extract.py`: PDF text extraction.
    - `format.py`: Text formatting.
  - `model/`: Model initialization and generation.
    - `__init__.py`
    - `initialize.py`: Model initialization.
    - `generate.py`: Text generation from the model.
  - `api/`: API and Streamlit app.
    - `__init__.py`
    - `app.py`: Streamlit app for interaction.
    - `api.py`: Flask/FastAPI app for API.

- `notebooks/`: Jupyter notebooks for testing and prototyping.
  - `initial_notebook.ipynb`: Your initial notebook for testing.

- `tests/`: Unit tests for your modules.
  - `__init__.py`
  - `test_extract.py`: Unit tests for text extraction.
  - `test_format.py`: Unit tests for text formatting.
  - `test_initialize.py`: Unit tests for model initialization.
  - `test_generate.py`: Unit tests for text generation.


## Prompt engineering : AUTOMAT framework

AUTOMAT is a framework used in prompt engineering to guide the creation of effective prompts for chatbots and AI models. It helps define the role and behavior of the AI, such as acting as a financial advisor when interacting with potential clients. This framework provides a structured approach to prompt design, ensuring that the AI's responses are tailored to specific tasks and user interactions.

Key aspects of the AUTOMAT framework include :
* **Act as** : Defining the role the AI should assume (e.g., financial advisor) ;
* **User** : Specifying the target audience (e.g., potential clients) ;
* **Task** : Outlining the specific task or goal for the interaction ;
* **Output** : Determining the desired format or type of response ;
* **Manner** : Establishing the tone and style of communication ;
* **Additional context** : Providing any extra information to enhance the AI's understanding.

By using AUTOMAT, prompt engineers can create more precise and effective prompts, leading to improved AI performance and more relevant responses for users.

## Others

```python
# Target a specific page from the resulting list of extracted pages
list(filter(lambda d: d.get('page_number') == 2, extracted_pages))
```
