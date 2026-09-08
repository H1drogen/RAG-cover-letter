from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

def create_agent(instructions: str, tools):

    # COMPANY_RESEARCHER_INSTRUCTIONS = """You research target companies using web search.
    #
    # For each company name provided, use web_search to find:
    # - Company mission, values, and culture
    # - Recent news, products, or services
    # - Company size, industry, and key achievements
    # - Team composition and leadership (if relevant to the role)
    #
    # Compile findings into a concise research summary with:
    # - Company overview and what they do
    # - Key differentiators and competitive advantages
    # - Recent developments or announcements
    # - Any information relevant to the job role
    #
    # Return insights in under 300 words. Use bullet points or short paragraphs for clarity. Avoid speculation; only include verified information from reputable sources."""
    #
    # chunk_analyst_subagent = {
    #     "name": "chunk-analyst",
    #     "description": (
    #         "Analyze one retrieved documentation chunk file. "
    #         "Pass the user question and a single file path under /retrieved/."
    #     ),
    #     "system_prompt": COMPANY_RESEARCHER_INSTRUCTIONS,
    # }

    chunk_analyst_subagent = {
        "name": "chunk-analyst",
        "description": "Analyze documentation chunks from RAG retrieval.",
        "system_prompt": """Analyze retrieved documentation chunks and extract key facts.""",
    }

    background_researcher = {
        "name": "company-background-researcher",
        "description": "Research company background, mission, and market position.",
        "system_prompt": """Research company background and fundamentals.

    Use web_search to find:
    - Company mission, vision, and values
    - Industry and market position
    - Key products/services overview related to role
    """,
    }

    news_researcher = {
        "name": "company-news-researcher",
        "description": "Research recent company news and developments.",
        "system_prompt": """Research company recent news and developments.

    Use web_search to find:
    - Recent announcements and press releases
    - Product launches or updates

    Focus on last 6-12 months. Return under 300 words."""
    }

    model = init_chat_model(model="openai:gpt-5.5")

    agent = create_deep_agent(
        model=model,
        system_prompt=instructions,
        tools=tools,
        subagents=[
            background_researcher,
            news_researcher
        ],
    )

    return agent