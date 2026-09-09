from pathlib import Path

from dotenv import load_dotenv

from deepagents.backends import StateBackend
from langchain_classic.chains.llm_summarization_checker.base import PROMPTS_DIR
from langchain_core.messages import HumanMessage

from generation.agents.agent import create_agent
from generation.tools.search import search_content
from src.indexing import *
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
    print(docs)

    retriever = create_retriever(docs, embedding)

    company_name = "Citadel"
    role_title = "Software Engineer – Intern (Europe)"
    job_description = r"""
    At Citadel, our engineers work in small teams to turn the best ideas into high-performing and resilient technology. With short development cycles, work rapidly goes into production. As an engineer, you can create system architectures, develop platforms and build web frameworks. You’ll have access to state-of-the-art tools and apply innovative techniques including distributed computing, natural language processing, machine learning and more.
    As an intern, you’ll get to challenge the impossible in technology through an 11-week program that will allow you to collaborate and connect with senior team members. In addition, you’ll get the opportunity to network and socialize with peers throughout the internship.
    Your Objectives:
    Create technological tools that bring trading strategies to life
    Develop high-performance, large data research platforms
    Work in small teams to build the future of finance
    Your Skills & Talents:
    Bachelor's, master's or PhD in computer science, computer engineering or related fields
    Exceptional programming and design skills
    Strong analytical skills and familiarity with probability and statistics
    Ability to communicate effectively in a collaborative, complex and highly technical team environment
    Intellectual curiosity and passion for solving challenging problems using technology
    """,

    retrieved_context = retriever.invoke(
        f"{job_description}\n\nFind the most relevant candidate experiences based on this job description, "
        "skills, projects, and achievements"
    )

    candidate_evidence = "\n".join([
        f"- {doc.page_content}..."
        for doc in (retrieved_context if isinstance(retrieved_context, list) else [retrieved_context])
    ])

    with open(f"output/RAG_context.txt", "w") as f:
        f.write(candidate_evidence)

    # The below tools cannot be created because the current architecture loads context immediately into context window, instead of allowing the agent to delegate and control chunking retrieval. This is a limitation of the current implementation and will be addressed in future iterations.
    # retriever_tool = retrieve_content
    # bound_search = search_content.bind(retriever=retriever)
    PROMPTS: dict[str, str] = {}

    for path in Path('src/generation/prompts/').iterdir():
        if not path.is_dir():
            var_name = path.name
            content = path.read_text(encoding="utf-8")
            PROMPTS[var_name] = content

    SYSTEM_PROMPT = (
        # + "\n\n"
        # + "=" * 80
        # + "\n\n" +
        PROMPTS["sys_cover_letter.txt"].format()
    )

    internet_search = {"type": "web_search"}
    agent = create_agent(SYSTEM_PROMPT, [internet_search])


    HUMAN_QUERY = PROMPTS["human_prompt.txt"].format(
        job_description=job_description,
        company_name=company_name,
        role_title=role_title,
        retrieved_context=candidate_evidence,
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content=HUMAN_QUERY)]}
    )

    for msg in result.get("messages", []):
        if "chunk-analyst" in str(msg.content):
            print("✅ Chunk analyst processed RAG results")
        if "/retrieved/" in str(msg.content):
            print(f"✅ RAG FILES USED:\n{msg.content}")
        if msg.text:
            print(msg.text)
            with open(f"output/cover_letter.txt", "w") as f:
                f.write(msg.text)

if __name__ == "__main__":
    main()