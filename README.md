# Multi-Agent Customer Support Crew (CrewAI + Streamlit)

A customer support app built with CrewAI and a Streamlit interface. Three agents run one after another (sequential process):

1. **Assistant** answers the query from its own knowledge.
2. **Web Search Assistant** searches the web (Serper) and answers from the results.
3. **Entry Agent** saves the query and both answers into `answers.txt`.

The Streamlit page shows both answers to the user.

## Requirements

- Python 3.10 to 3.13
- OpenAI API key
- Serper API key (serper.dev)

## Setup (Windows)

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your keys:

```
OPENAI_API_KEY=your-openai-key-here
SERPER_API_KEY=your-serper-key-here
```

## Run

```
streamlit run app.py
```

The app opens at http://localhost:8501

## Security

Never upload your `.env` file. It is listed in `.gitignore`.