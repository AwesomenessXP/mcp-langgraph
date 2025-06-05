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
	•	certificate_type: Certificate type is specified (e.g., ACORD 25).
    •	certificate_holder: Certificate holder is correctly named.
    •	producer: Subcontractor's insurance agent (name and location).
    •	insured: Subcontractor's legal name and business address.
    •	carriers: All insurance providers for each policy.
	•	coverage_types_present: All required coverage types are included.
	•	coverage_limits: Coverage limits are provided for each applicable policy.
	•	policy_dates: Effective and expiration dates are listed for all coverages.
    •	policy_numbers: Policy numbers are listed for all coverages.
	•	project_identification: Project name or address is included for identification.
	•	additional_insureds: All required additional insured entities are listed; note any missing.
	•	waiver_of_subrogation: Waiver of subrogation is present where required.
	•	primary_and_noncontributory: Primary and noncontributory wording is included where required.
	•	endorsements: Required endorsements (for each coverage type) are present and wording is correct.
	•	additional_endorsements: Additional endorsements (e.g., primary and noncontributory, waiver of subrogation for GL) are included.
	•	missing_fields: Any required fields that could not be found are listed, with the reason.

    Additional Endorsements (as above, with correct wording if possible)

    Do not add any explanation or formatting. Output ONLY the JSON object.
    """
    prompt = PROMPT_TEMPLATE.format(document=document)
    # You can swap out with your LLM of choice, or use LangChain for abstraction
    response = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

@mcp.tool()
def analyze_summary(summary: str) -> str:
    """
    You are an ACORD 25 insurance expert. Analyze extracted summary for compliance.
    """
    
    ANALYSIS_PROMPT = """
    Given the following extracted insurance summary as a JSON object:

    {summary}

    Analyze for the following and respond with analysis for each key with ONLY the JSON object with the following keys:

    1. Basics:
        •	producer: Subcontractor's insurance aagent's name and location.
        •	insured: Subcontractor's legal name and business address.
        •	carriers: All insurance providers for each policy.
        •	Policy Details: Policy number and expiration date for:
            •	General Liability
            •	Auto Liability
            •	Umbrella/Excess Liability
            •	Workers's Compensation
            •	Professional, Pollution, or Inland Marine (if applicable)

    ⸻

    2. Minimum Coverage Limits (Meet or Exceed):
        •	(General Liability, Auto Liability, Umbrella/Excess Liability, Workers's Compensation): $1M per occurrence
        •	Professional/Pollution/Inland Marine (if applicable): $2M per occurrence

    ⸻

    3. General Liability:
        •	Project Box: Must be checked
        •	OR the CG 25 03 05 09 Per Project Aggregate endorsement must be attached.

    ⸻

    4. Specialty Coverages:
        •	Professional, Pollution, and Inland Marine:
        •	“Other Insurance” section must reference these coverages as applicable.
        •	Professional Liability: Required for any design or testing services (does not need to be project-specific).
        •	Pollution/Inland Marine: Must be project-specific.

    ⸻

    5. Description of Operations (Must Include All):
        •	Job Name
        •	Job Address
        •	Simile Construction Project #
        •	“Simile Construction Service, Inc.” is listed
        •	All required additional insureds per subcontract agreement

    ⸻

    6. Certificate Holder (Must Match Exactly):

    Simile Construction Service, Inc.
    4725 Enterprise Way #1
    Modesto, CA 95356

    ⸻

    7. Required Endorsements for General Liability (Must Remain Valid Through Project Warranty Period):
        •	CG 20 10 07 04 - Ongoing operations
        •	Must state either:
        •	“As required by written contract/agreement”
        •	OR: “Simile Construction Service, Inc., its directors, officers, and employees and any other person or organization as required by written contract”
        •	CG 20 37 07 04 - Completed operations
        •	Must include same additional insured language as CG 20 10.
        •	CG 20 01 04 13 - Primary & Non-Contributory
        •	Must be a separate endorsement OR listed in Description.
        •	CG 24 04 05 09 - Waiver of Subrogation
        •	Protects Simile Construction from liability claims by the subcontractor's insurer.
        •	CG 25 03 05 09 - Per Project Aggregate
        •	Required if Project box is not checked on the COI.

    ⸻

    8. Auto Insurance Endorsements (All Must Be Included):
        •	Additional Insured
        •	Primary Wording
        •	Waiver of Subrogation

    ⸻

    9. Workers's Compensation Endorsement:
        •	WC 00 03 13 - Waiver of Subrogation:
            •	Must list:
                •	Simile Construction Service, Inc.
                •	The Project Owner and Client
                •	Anyone else required by written contract
    """

    prompt = ANALYSIS_PROMPT.format(summary=summary)
    response = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
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
    - Address the certificate holder by name.
    - Briefly summarize the analysis/recommendation.
    - Sign off as 'Insurance Team'.

    Format the output as a professional email with a subject line, greeting, body, and closing, all as a single string.
    Do not use markdown or code blocks—output only the email as it would be sent.
    """
    prompt = EMAIL_PROMPT.format(analysis=analysis)
    response = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
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