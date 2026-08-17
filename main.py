from dotenv import load_dotenv

from deepagents.backends import StateBackend
from langchain_classic.chains.llm_summarization_checker.base import PROMPTS_DIR
from langchain_core.messages import HumanMessage

from agent.generation.agent import create_agent
from agent.tools.search import search_content
from indexing.ingest import load_context_docs, split_docs
from indexing.embed import get_embeddings, EmbeddingConfig
from indexing.embed import create_retriever, retrieve_content
import os

def main():
    load_dotenv()
    # os.environ("LANGSMITH_API_KEY")

    config = EmbeddingConfig(
        'openai',
        openai_api_key= os.getenv('OPENAI_API_KEY'),
        openai_base_url= os.getenv('OPENAI_BASE_URL'),
        openai_model= os.getenv('OPENAI_MODEL'),

        bedrock_region_name= os.getenv('BEDROCK_REGION'),
        bedrock_profile_name= os.getenv('BEDROCK_PROFILE'),
        bedrock_model_id=os.getenv('BEDROCK_MODEL_ID')
    )
    embedding = get_embeddings(config)

    docs = load_context_docs("data/")
    docs = split_docs(docs)
    print(docs.content)

    retriever = create_retriever(docs, embedding)

    PROMPTS: dict[str, str] = {}

    for path in sorted(PROMPTS_DIR.glob("*.txt")):
        var_name = path.name
        content = path.read_text(encoding="utf-8")
        PROMPTS[var_name] = content

    INSTRUCTIONS = (
        PROMPTS["cover_letter.txt"]
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + PROMPTS["subagent_delegation.txt"].format(
            max_concurrent_analysts=3,
        )
    )

    EXAMPLE_QUERY = "How do I stream intermediate tool results from a subagent?"
    backend = StateBackend()

    agent = create_agent(INSTRUCTIONS, [search_content], backend)

    result = agent.invoke(
        {"messages": [HumanMessage(content=EXAMPLE_QUERY)]}
    )

    for msg in result.get("messages", []):
        if msg.text:
            print(msg.text)

if __name__ == "__main__":
    main()