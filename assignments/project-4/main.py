import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    RunContextWrapper,
    function_tool,
    set_tracing_disabled
)

set_tracing_disabled(disabled=True)

# Load environment
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Initialize OpenAI client
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# ============ Context Schema ===========
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    is_premium: bool
    issue_type: str  # technical, billing, refund

# ============ Tools ====================

@function_tool
async def check_account_status(ctx: RunContextWrapper):
    user = ctx.inputs["user_info"]
    if user.is_premium:
        return f"{user.name} is a premium user. Handle with priority. 🎖️"
    else:
        return f"{user.name} is a standard user."

@function_tool
async def check_refund_policy(ctx: RunContextWrapper):
    def is_enabled(ctx: RunContextWrapper) -> bool:
        user = ctx.inputs["user_info"]
        return user.is_premium  # Only enabled for premium users
    check_refund_policy.is_enabled = is_enabled
    return "Refunds are only processed within 7 days of purchase, and only for major issues."

@function_tool
async def restart_service(ctx: RunContextWrapper):
    def is_enabled(ctx: RunContextWrapper) -> bool:
        user = ctx.inputs["user_info"]
        return user.issue_type == "technical"  # Only enabled for technical issues
    restart_service.is_enabled = is_enabled
    return "Service has been restarted. Please check if the issue is resolved."

@function_tool
async def general_faq(ctx: RunContextWrapper):
    return "Please visit our FAQ page at www.example.com/faq for common questions and answers."

@function_tool
async def triage(ctx: RunContextWrapper) -> str:
    user = ctx.inputs["user_info"]
    if user.issue_type == "billing" or user.issue_type == "refund":
        return "BillingAgent"
    elif user.issue_type == "technical":
        return "TechnicalAgent"
    else:
        return "GeneralSupportAgent"

# ============ Agents ====================

BillingAgent = Agent(
    name="BillingAgent",
    instructions="You are a billing support expert. Handle all billing-related issues and explain policies clearly.",
    tools=[check_account_status, check_refund_policy],
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)
)

TechnicalAgent = Agent(
    name="TechnicalAgent",
    instructions="You are a tech support engineer. Help with technical issues in a simple, polite way.",
    tools=[check_account_status, restart_service],
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)
)

GeneralSupportAgent = Agent(
    name="GeneralSupportAgent",
    instructions="You handle all general support questions. Be polite and helpful.",
    tools=[general_faq],
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)
)

SupportTriageAgent = Agent(
    name="SupportTriageAgent",
    instructions="You decide which support agent should handle a query based on issue_type. Use the triage tool.",
    tools=[triage],
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)
)

# ============ Runner Workaround =====================

# Since Runner doesn't accept arguments, we'll simulate handoff manually
async def run_agent(agent_name: str, user_query: str, context: dict):
    agent_map = {
        "BillingAgent": BillingAgent,
        "TechnicalAgent": TechnicalAgent,
        "GeneralSupportAgent": GeneralSupportAgent,
        "SupportTriageAgent": SupportTriageAgent
    }
    agent = agent_map.get(agent_name)
    if not agent:
        return f"Error: Agent {agent_name} not found."

    runner = Runner()  # Empty initialization
    result = await runner.run(
        agent_name=agent_name,
        input=user_query,
        context=context
    )
    return result.output

# ============ Entry Point =================

async def main():
    print("🤖 Welcome to the Console Support Agent System!\n")

    while True:
        try:
            # Ask for user context
            user_name = input("👤 Enter your name: ").strip()
            issue_type = input("🔧 Enter issue type (technical / billing / refund): ").strip().lower()
            premium_input = input("💎 Are you a premium user? (yes / no): ").strip().lower()
            is_premium = premium_input in ["yes", "y"]

            user_data = UserInfo(
                name=user_name,
                is_premium=is_premium,
                issue_type=issue_type
            )

            # Ask for user query
            user_query = input("📝 What's your question? ").strip()

            # Run the triage agent first
            triage_result = await run_agent(
                agent_name="SupportTriageAgent",
                user_query=user_query,
                context={"user_info": user_data}
            )

            # Get the target agent from triage
            target_agent = triage_result
            print(f"🔄 Triage decided to hand off to: {target_agent}")

            # Run the target agent
            final_result = await run_agent(
                agent_name=target_agent,
                user_query=user_query,
                context={"user_info": user_data}
            )

            print(f"\n💬 AI Response: {final_result}\n")

            cont = input("🔁 Do you want to ask another question? (yes / no): ").strip().lower()
            if cont not in ["yes", "y"]:
                print("👋 Goodbye!")
                break

        except Exception as e:
            print(f"❌ Error: {e}")
            break

if __name__ == "__main__":
    asyncio.run(main())
