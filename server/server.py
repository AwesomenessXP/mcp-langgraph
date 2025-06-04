from fastmcp import FastMCP
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# math_server.py
mcp = FastMCP("insurance_compliance")

import openai

# Instantiate the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@mcp.tool()
def extract_summary(document: str) -> str:
    """
    You are an ACORD 25 insurance expert. Extract key summary info from the insurance JSON string via LLM.
    """

    PROMPT_TEMPLATE = """
    Given the following insurance document in JSON format:

    {document}

    1. Format the document as a JSON object.

    2. Extract and output a JSON object with the following keys:
    - coverage_types (GL, Auto, Workers' Comp, Prof. Liability, Umbrella Liability)
    - coverage_limits (stated for GL, Auto, Workers' Comp, Prof. Liability, Umbrella Liability, check $1M minimum)
    - effective_date for each coverage type
    - expiry_date for each coverage type
    - named_insured (exact match needed)
    - certificate_holder (“Simile Construction” and others)
    - project_id (any present)
    - additional_insureds (“Simile Construction”, owner, client; list missing)
    - waiver_of_subrogation (flag presence)
    - endorsements (CG 20 10, CG 20 37, equivalents, Workers's Comp endorsements per needs)
    Additional Endorsements (as above, with correct wording if possible)


    Do not add any explanation or formatting. Output ONLY the JSON object.
    """
    prompt = PROMPT_TEMPLATE.format(document=document)
    # You can swap out with your LLM of choice, or use LangChain for abstraction
    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

@mcp.tool()
def analyze_summary(summary: str) -> str:
    """
    You are an ACORD 25 insurance expert. Analyze extracted summary for issues.
    Returns a recommendation message string.
    """
    ANALYSIS_PROMPT = """
    Given the following extracted insurance summary as a JSON object:

    {summary}

    Analyze for the following and espond with output ONLY the JSON object with the following keys:
    - producer (subcontractor's insurance agent (name and location))
    - insured (subcontractor's legal name and business address)
    - carriers (all insurance providers must be named)
    - policy_details (confirm policy numbers and expiration dates for each coverage)
    - general_liability (project box must be checked unless CG 25 03 05 09 endorsement is provided)
    - specialty_coverages (“Other Insurance” section: Used for Professional Liability: Required for any design or testing services. Does not need to be project-specific.)
    - description_of_operations (must include all of the following)
    - certificate_holder (must be listed exactly)
    - required_endorsements (must remain valid through the warranty period of the project)
    - auto_insurance_endorsements (must include: Additional Insured, Primary Wording, Waiver of Subrogation)
    - workers_compensation_endorsement (must list: The Project Owner and Client, or anyone else required by written contract)
    - additional_endorsements (as above, with correct wording if possible)
    - limits for each coverage type (check $1M minimum)
    """

    prompt = ANALYSIS_PROMPT.format(summary=summary)
    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()

@mcp.tool()
def format_email(analysis: str) -> str:
    """
    You are an ACORD 25 insurance expert. Compose a personalized email to the given certificate holder, using extracted summary and analysis.
    """
    EMAIL_PROMPT = """

    And the following analysis/recommendation:
    {analysis}

    Write a professional, clear email to the policyholder.
    - Address them by name.
    - Include the policy holder's name.
    - Briefly summarize the analysis/recommendation.
    - Sign off as 'Insurance Team'.

    Only output the email as a paragraph as a string.
    """
    prompt = EMAIL_PROMPT.format(analysis=analysis)
    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

@mcp.tool()
def divide(a: int, b: int) -> int:
    """Divide two numbers"""
    answer = a / b
    raise Exception("Unrecoverable error for testing purposes.")

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8001)