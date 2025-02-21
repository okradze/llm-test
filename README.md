# LLM Test

## Result

View the `result.json` file [here](https://github.com/okradze/llm-test/blob/main/result.json)

## Setup

Requirements:

-   Python >= 3.10
-   [OpenRouter](https://openrouter.ai/) API KEY

Create `.env` file from `.env.example` and add your OpenRouter API KEY.

Create virtual env

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies and run the script to evaluate the models.

```bash
pip install -r requirements.txt
python3 main.py
```
