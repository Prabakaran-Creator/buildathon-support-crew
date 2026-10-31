import os

# Turn off CrewAI's "share trace" question so the web page does not wait for it
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from crewai_tools import SerperDevTool

# Read the API keys from the .env file into environment variables
load_dotenv()


# Tool for the Entry Agent: writes everything into answers.txt
@tool("Save Entry")
def save_entry(query: str, answer_1: str, answer_2: str) -> str:
    """Save the query and both answers into the answers.txt file."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("answers.txt", "a", encoding="utf-8") as file:
        file.write(f"Time: {now}\n")
        file.write(f"Query: {query}\n")
        file.write(f"Answer 1 (Assistant): {answer_1}\n")
        file.write(f"Answer 2 (Web Search): {answer_2}\n")
        file.write("-" * 60 + "\n")
    return "Saved to answers.txt"


# Read both answers from the Entry Agent's reply
def split_answers(text):
    text = str(text)
    if "ANSWER 1:" in text and "ANSWER 2:" in text:
        part_1 = text.split("ANSWER 1:")[1].split("ANSWER 2:")[0].strip()
        part_2 = text.split("ANSWER 2:")[1].strip()
        return part_1, part_2
    return None, None


# Build the crew and run it for one query
def run_crew(query):
    # Today's date, so the web search stays current
    today = datetime.now().strftime("%B %d, %Y")
    month_year = datetime.now().strftime("%B %Y")

    # Agent 1: answers from its own knowledge
    assistant = Agent(
        role="Assistant",
        goal="Answer the customer's query directly from your own knowledge.",
        backstory="You are a friendly customer support assistant.",
        llm="gpt-4o-mini",
        verbose=False,
    )

    # Agent 2: searches the web (only this agent has the search tool)
    web_assistant = Agent(
        role="Web Search Assistant",
        goal="Search the web and answer the customer's query using only the latest search results.",
        backstory=(
            "You are a careful researcher. You never answer from memory because "
            "your memory may be outdated. You only use what the web search returns."
        ),
        tools=[SerperDevTool()],
        llm="gpt-4o-mini",
        verbose=True,
    )

    # Agent 3: saves the record in a text file
    entry_agent = Agent(
        role="Entry Agent",
        goal="Save the query and both answers into a text file.",
        backstory="You are an organised assistant who keeps accurate records.",
        tools=[save_entry],
        llm="gpt-4o-mini",
        verbose=False,
    )

    task_1 = Task(
        description=(
            "Customer query: {query}\n"
            "Answer it directly from your own knowledge. "
            "Write 3 to 4 short, friendly sentences in plain text."
        ),
        expected_output="A short, friendly plain-text answer.",
        agent=assistant,
    )

    task_2 = Task(
        description=(
            "Today's date is {today}.\n"
            "Customer query: {query}\n"
            "You MUST use the search tool. Add the words '{month_year}' to your search "
            "so the results are recent. Do NOT answer from your own memory. "
            "Write your answer using ONLY the search results, and prefer the most "
            "recent information. If the results show dates, trust the newest ones. "
            "Start your answer with 'As of {today},'. "
            "Write 3 to 4 short, friendly sentences in plain text. No links."
        ),
        expected_output="A short, friendly plain-text answer based on the latest web search.",
        agent=web_assistant,
    )

    # The Entry Agent receives both earlier answers through context
    task_3 = Task(
        description=(
            "Customer query: {query}\n"
            "Use the Save Entry tool exactly once to save the query, "
            "answer 1 (from the Assistant) and answer 2 (from the Web Search Assistant). "
            "Do not change any word of the answers. "
            "After saving, reply in EXACTLY this format:\n"
            "ANSWER 1:\n<answer 1 text>\n"
            "ANSWER 2:\n<answer 2 text>"
        ),
        expected_output="The two answers in the format ANSWER 1 and ANSWER 2.",
        agent=entry_agent,
        context=[task_1, task_2],
    )

    # Sequential process: the agents run one after another
    crew = Crew(
        agents=[assistant, web_assistant, entry_agent],
        tasks=[task_1, task_2, task_3],
        process=Process.sequential,
    )
    result = crew.kickoff(
        inputs={"query": query, "today": today, "month_year": month_year}
    )

    answer_1, answer_2 = split_answers(result)

    # Backup: if the format was not followed, use the task outputs directly
    if answer_1 is None:
        answer_1 = task_1.output.raw
        answer_2 = task_2.output.raw

    return answer_1, answer_2


# ---------------- Streamlit page ----------------
st.set_page_config(page_title="Support Crew")
st.title("Customer Support Crew")
st.write("Ask a question. Two AI agents answer it, and a third agent saves the record.")

query = st.text_input("Your question or task")

if st.button("Get Answer"):
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("SERPER_API_KEY"):
        st.error("API keys not found. Please check your .env file.")
    elif query.strip() == "":
        st.warning("Please type a question first.")
    else:
        with st.spinner("Our agents are working on your question..."):
            answer_1, answer_2 = run_crew(query.strip())

        st.subheader("Answer 1: Assistant")
        st.write(answer_1)

        st.subheader("Answer 2: Web Search Assistant")
        st.write(answer_2)

        st.success("Saved to answers.txt")