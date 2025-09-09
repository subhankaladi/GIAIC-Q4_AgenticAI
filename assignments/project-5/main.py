# Import core classes and decorators from the OpenAI Agents SDK
from agents import (
    Agent,                       # Defines an AI agent with instructions, tools, and guardrails
    Runner,                      # Handles running agents with given inputs/configurations
    RunContextWrapper,           # Wraps contextual state passed between runs/guardrails/tools
    GuardrailFunctionOutput,     # Standard return type for guardrail functions
    input_guardrail,             # Decorator for defining input guardrail functions
    output_guardrail,            # Decorator for defining output guardrail functions
    RunConfig,                   # Configuration object for model run (model, tracing, etc.)
    AsyncOpenAI,                 # Async OpenAI API client
    OpenAIChatCompletionsModel,  # OpenAI Chat Completions model wrapper
    function_tool,               # Decorator for defining callable tools
    set_tracing_disabled         # Disable tracing
)

# Standard library imports
import os                       # For accessing environment variables
from dotenv import load_dotenv  # To load environment variables from .env file
from pydantic import BaseModel  # For defining structured data models
import asyncio                  # For running async code in main()
import json                     # For parsing/encoding JSON data

# Disable tracing
set_tracing_disabled(disabled=True)

# Load environment variables from .env file
load_dotenv()

# Read Gemini API key from environment
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Create async OpenAI API client
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Define the model for all agents
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

# Global run configuration
config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

# Pydantic Models
class UserInfo(BaseModel):
    """Represents a user's bank account details."""
    name: str
    account_number: str
    balance: float
    pin: int

class InputCheck(BaseModel):
    """Used by input guardrail to check if a query is bank-related."""
    is_bank_related: bool

# Agents for banking operations
deposit_agent = Agent(
    name="Deposit Agent",
    instructions="""You are a deposit agent. Answer questions about making deposits. 
    Return answers in plain text or JSON when appropriate.""",
    model=model,
)

withdrawal_agent = Agent(
    name="Withdrawal Agent",
    instructions="You are a withdrawal agent. Answer questions about making withdrawals.",
    model=model,
)

balance_agent = Agent(
    name="Balance Agent",
    instructions="You are a balance agent. If asked for balance, call the get_user_info tool and return a short answer.",
    model=model,
)

# Guardrail Agent for input filtering
input_guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions="""You are a guardrail. Given the user's text, return JSON with a boolean key 
    'is_bank_related' set to true if the user asks about banking/accounts/payments/etc., 
    otherwise false. Return only JSON, e.g. {"is_bank_related": true}.""",
    output_type=InputCheck,
    model=model,
)

# Input guardrail function
@input_guardrail
async def banking_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    user_input: str
) -> GuardrailFunctionOutput:
    """Checks if the user input is banking-related. Blocks non-banking inputs."""
    res = await Runner.run(
        input_guardrail_agent, input=user_input, context=ctx.context, run_config=config
    )
    final = getattr(res, "final_output", None)
    is_bank = False

    if isinstance(final, dict):
        is_bank = bool(final.get("is_bank_related", False))
    elif final is not None and hasattr(final, "is_bank_related"):
        try:
            is_bank = bool(getattr(final, "is_bank_related"))
        except Exception:
            is_bank = False
    elif isinstance(final, str):
        try:
            parsed = json.loads(final)
            if isinstance(parsed, dict):
                is_bank = bool(parsed.get("is_bank_related", False))
            else:
                is_bank = "true" in final.lower()
        except Exception:
            is_bank = any(keyword in user_input.lower() for keyword in ["bank", "balance", "deposit", "withdrawal", "account"])

    return GuardrailFunctionOutput(output_info=user_input, tripwire_triggered=not is_bank)

# Guardrail Agent for output filtering
output_guardrail_agent = Agent(
    name="Guardrail Output Agent",
    instructions="""You are a guardrail for outputs. Ensure the text does not leak sensitive info like PIN. 
    If you detect sensitive details, respond with a safe refusal message. Otherwise return the original response.""",
    model=model,
)

# Output guardrail function
@output_guardrail
async def output_guardrail_fn(
    ctx: RunContextWrapper[None],
    agent: Agent,
    output: str
) -> GuardrailFunctionOutput:
    """Checks output for sensitive info like PIN."""
    res = await Runner.run(
        output_guardrail_agent, input=output, context=ctx.context, run_config=config
    )
    final = getattr(res, "final_output", None)

    safe_output = final
    if isinstance(final, dict):
        safe_output = json.dumps(final)
    elif final is None:
        safe_output = ""
    else:
        safe_output = str(final)

    return GuardrailFunctionOutput(output_info=safe_output, tripwire_triggered=False)

# Tool: Retrieve user info
user_data = UserInfo(name="shahid", account_number="42338734", balance=10000.0, pin=9876)

@function_tool
async def get_user_info(ctx: RunContextWrapper[None]) -> dict:
    """Returns user info as a dictionary for the model to consume."""
    return {
        "user_name": user_data.name,
        "account_no": user_data.account_number,
        "response": f"Your current balance is ${user_data.balance:.2f}",
    }

# Main Agent
main_agent = Agent(
    name="Bank Agent",
    instructions="""You are a helpful bank agent. Follow these rules:
    - If the user asks about deposits -> handoff to Deposit Agent.
    - If the user asks about withdrawals -> handoff to Withdrawal Agent.
    - If the user asks about account balance or account details -> call the get_user_info tool.
    - Always return concise answers. If returning structured info, return JSON with keys:
      user_name, account_no, response.""",
    model=model,
    handoffs=[deposit_agent, withdrawal_agent, balance_agent],
    tools=[get_user_info],
    input_guardrails=[banking_guardrail],
    output_guardrails=[output_guardrail_fn],
)

# Runner
async def main():
    try:
        # Example 1: Balance query (should call get_user_info)
        result = await Runner.run(
            main_agent,
            input="what is my current balance?",
            # input="what is the current weather in karachi?",
            run_config=config,
        )
        print("== Result FINAL OUTPUT ==")
        print(result.final_output)
        print("== Type of final_output ==", type(result.final_output))

        # Example 2: Deposit query (should handoff to Deposit Agent)
        result2 = await Runner.run(
            main_agent,
            input="i have to deposite but i forget my pin,plz help.",
            run_config=config,
        )
        print("== Result2 FINAL OUTPUT ==")
        print(result2.final_output)
        print("== Type ==", type(result2.final_output))

    except Exception as e:
        print(f"An error occurred: {e}")

# Entry point
if __name__ == "__main__":
    asyncio.run(main())
