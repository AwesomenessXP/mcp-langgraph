from fastmcp import FastMCP
import json
from datetime import datetime
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
    Extract key summary info from the insurance JSON string via LLM.
    """

    PROMPT_TEMPLATE = """
    Given the following insurance document in JSON format:

    {document}

    Extract and output a JSON object with the following keys:
    - policy_number
    - holder_name
    - holder_age
    - num_claims
    - last_claim_date (or null)
    - coverage_total
    - email_address

    Do not add any explanation or formatting. Output ONLY the JSON object.
    """
    prompt = PROMPT_TEMPLATE.format(document=document)
    # You can swap out with your LLM of choice, or use LangChain for abstraction
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

@mcp.tool()
def analyze_summary(summary: str) -> str:
    """
    Analyze extracted summary for issues.
    Returns a recommendation message string.
    """
    ANALYSIS_PROMPT = """
    Given the following extracted insurance summary as a JSON object:

    {summary}

    Analyze for the following:
    - Is the coverage total below recommended levels (below 200,000)?
    - Are there recent claims (num_claims > 0)? List the date if so.
    - Give clear, actionable recommendations or say "Your policy looks good. No issues found."

    Respond with output ONLY the JSON object with the following keys:
    - policy_number
    - analysis
    - recommendation
    - contact_support
    - email_address
    """

    prompt = ANALYSIS_PROMPT.format(summary=summary)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()

@mcp.tool()
def format_email(analysis: str) -> str:
    """
    Compose a personalized email to the given email address, using extracted summary and analysis.
    """
    EMAIL_PROMPT = """
    You are an insurance assistant.

    And the following analysis/recommendation:
    {analysis}

    Write a professional, clear email to the policyholder.
    - Address them by name.
    - Include their policy number.
    - Briefly summarize the analysis/recommendation.
    - Sign off as 'Insurance Team'.


    Only output the email as a string.
    """
    prompt = EMAIL_PROMPT.format(analysis=analysis)
    response = client.chat.completions.create(
        model="gpt-4o",
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