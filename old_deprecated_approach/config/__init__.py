import yaml
from langchain.chat_models import init_chat_model


# llm
def load_llm():
    return init_chat_model("anthropic:claude-3-5-haiku-latest")


# functions
def load_functions(path="prompts.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
