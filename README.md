# Local RAG for GCP Professional Data Engineer certification

This project aims at building a local RAG which will help in training for the Google Cloud Cloud Professional Data Engineer certification by generating exam questions.

Various topics will be covered on the journey to build this RAG like :
* extracting content from a PDF file ;
* embeddings text ;
* generating output based on retrieved context ;
* creating custom prompt templates ;
* building a user interface.

> **Note**  
> &#x1F64F; This project won't have been possible without the great video tutorial ([Local Retrieval Augmented Generation (RAG) from Scratch](#https://youtu.be/qN_2fnOPY-M?si=9dsfcNGMjgQhF8Bs)) from [Daniel Bourke](#https://www.mrdbourke.com/).

## Project structure


```
.
├── .gitignore
├── README.md
├── notebooks
│   └── rag-building-discovery.ipynb
├── notes.md
├── pdf
│   └── source.pdf
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── generation
│   │   ├── __init__.py
│   │   ├── augment_prompt.py
│   │   ├── generation_pipeline.py
│   │   ├── load_model.py
│   │   └── text_retriever.py
│   ├── helpers
│   │   └── timing_functions.py
│   └── preprocessing
│       ├── pdf_extractor.py
│       ├── preprocessing_pipeline.py
│       └── text_embedder.py
└── templates
    ├── generate_exam_question.yaml
    └── question_answer.yaml
```
