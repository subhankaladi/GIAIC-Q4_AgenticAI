import os
import asyncio
from typing import Optional
from pydantic import BaseModel, Field
from agents import (
    Agent,
    Runner,
    RunContextWrapper,
    input_guardrail,
    function_tool,
    GuardrailFunctionOutput,
    ModelSettings,
    RunConfig,
    OpenAIChatCompletionsModel,
    set_tracing_disabled
)
from openai import AsyncOpenAI
from dotenv import load_dotenv


# Disable tracing
set_tracing_disabled(disabled=True)

# ------------- Load API Key -------------
load_dotenv()

# Read Gemini API key from environment
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")


# ------------- OpenAI client & model -------------
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Define the model for all agents
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

# ModelSettings used where supported (Agent / RunConfig)
model_settings = ModelSettings(temperature=0.0)

run_config = RunConfig(
    model=model,
    model_settings=model_settings,
    tracing_disabled=True,
    model_provider=client,
)

# ------------- Book database -------------
# BOOK_DATABASE = {
#     "python programming": 5,
#     "data science handbook": 2,
#     "machine learning basics": 0,
#     "ai for everyone": 3,
#     "web development with django": 4,
# }

BOOK_DATABASE = {
    "python programming": 5,
    "deep learning with pytorch": 3,
    "natural language processing": 4,
    "computer vision essentials": 2,
    "big data analytics": 6,
}

# ------------- Pydantic User Context -------------
class UserContext(BaseModel):
    name: str
    member_id: Optional[str] = Field(default=None, description="Library member ID (None if not a member)")

# ------------- Guardrail Agent -------------
# This agent is a simple classifier that helps decide whether a query is library-related.
guardrail_agent = Agent(
    name="GuardrailAgent",
    instructions=(
        "You are a guardrail that decides whether a user's input is related to library services. "
        "Return a short plain answer containing words like 'library', 'book', 'availability', or 'timing' "
        "if it's library-related; otherwise return text indicating 'not library'."
    ),
    model=model,
    model_settings=model_settings,
)

# ------------- Input Guardrail (uses GuardrailAgent) -------------
@input_guardrail
async def library_input_guardrail(
    ctx: RunContextWrapper,
    agent: Agent,
    user_input: str | list[dict],
) -> GuardrailFunctionOutput:
    """
    Use the guardrail agent to classify queries. This guardrail will NOT raise a tripwire exception;
    instead it returns classification info so the library agent can refuse politely when needed.
    """
    # Run guardrail agent to classify
    result = await Runner.run(guardrail_agent, user_input, context=ctx)
    text = ""
    if result is not None:
        # Some SDK variants expose .final_output or .output_text — handle both
        text = getattr(result, "final_output", None) or getattr(result, "output_text", None) or str(result)
        text = str(text).lower()

    is_library = any(k in text for k in ("library", "book", "availability", "timing", "timings", "open", "hours"))

    # Don't trip the run; we'll let the main agent politely refuse. Return classification info.
    return GuardrailFunctionOutput(output_info={"classification": text or "unknown"}, tripwire_triggered=False)


# ------------- Function Tools -------------
@function_tool
def search_book_tool(query: str) -> str:
    """Search for a book in the BOOK_DATABASE (case-insensitive substring match)."""
    matches = [title for title in BOOK_DATABASE.keys() if query.lower() in title.lower()]
    if matches:
        pretty = ", ".join([m.title() for m in matches])
        return f"✅ Found books: {pretty}."
    return "❌ Book not found in the library."


@function_tool
def check_availability_tool(ctx: RunContextWrapper[UserContext], query: str) -> str:
    """Check availability for registered members only. Access user via ctx.context."""
    user: Optional[UserContext] = getattr(ctx, "context", None)
    # Some SDK variants store user in ctx.context; handle fallback for dict-like ctx
    if user is None:
        # Try to access via attribute on wrapper
        try:
            user = ctx.context  # type: ignore
        except Exception:
            user = None

    if not user or not getattr(user, "member_id", None):
        return "⚠️ You must be a registered member to check book availability. Provide a valid member_id."

    matches = [(book, copies) for book, copies in BOOK_DATABASE.items() if query.lower() in book.lower()]
    if not matches:
        return "❌ Book not found in the library."

    results = []
    for book, copies in matches:
        if copies > 0:
            results.append(f"📚 '{book.title()}' is available — {copies} copies.")
        else:
            results.append(f"🚫 '{book.title()}' is currently not available.")
    return "\n".join(results)


@function_tool
def get_library_timings() -> str:
    return "🕒 Library timings: Open from 9:00 AM to 5:00 PM, Monday to Friday."

# ------------- Dynamic instructions (personalization) -------------
def dynamic_instructions(context: RunContextWrapper[UserContext], agent: Agent) -> str:
    """
    A callable used as the Agent's instructions parameter to personalize the system prompt.
    This avoids relying on a removed `dynamic_instruction` import.
    """
    name = "Guest"
    try:
        name = getattr(context, "context", None).name  # type: ignore
    except Exception:
        try:
            name = getattr(context, "user", "Guest")
        except Exception:
            name = "Guest"
    return f"Hello {name}! You are a friendly Library Assistant. Help the user search books, check availability for members, and give timings. Refuse non-library questions politely."

# ------------- Library Agent -------------
library_agent = Agent(
    name="LibraryAgent",
    instructions=dynamic_instructions,  # pass the callable — dynamic personalization
    model=model,
    model_settings=model_settings,
    tools=[search_book_tool, check_availability_tool, get_library_timings],
    input_guardrails=[library_input_guardrail],
)

# ------------- RunContextWrapper (typed) -------------
class LibraryRunContext(RunContextWrapper[UserContext]):
    """Thin typed wrapper so we can attach user context object to runs."""
    pass


# ------------- Helper to print RunResult safely -------------
def _extract_response_text(run_result) -> str:
    if run_result is None:
        return ""
    # Try common attributes
    return (
        getattr(run_result, "final_output", None)
        or getattr(run_result, "output_text", None)
        or getattr(run_result, "final_response", None)
        or str(run_result)
    )

# ------------- Async tests using Runner.run -------------
async def async_tests():
    print("\n=== Async runs using Runner.run ===\n")
    # Registered member
    alice = UserContext(name="Alice", member_id="12345")
    ctx_alice = LibraryRunContext(alice)

    test_queries = [
        "Search for Python Programming",  # should trigger search
        "Check availability of Deep Learning With Pytorch ",  # member allowed (0 copies)
        "What are the library timings?",  # timings
        "Search for Natural Language Processing and check if it's available",  # multiple tools
        "What's the weather today?",  # non-library -> should be politely refused
    ]

    for q in test_queries:
        print(f"🔍 User Query: {q}")
        # result = await Runner.run(library_agent, q, context=ctx_alice, config=run_config)
        result = await Runner.run(library_agent, q, context=ctx_alice)
        print("💬 Agent Response:", _extract_response_text(result), "\n")

    # Non-member test
    bob = UserContext(name="Bob", member_id=None)
    ctx_bob = LibraryRunContext(bob)
    non_member_query = "Check availability of Computer Vision Essentials"
    print(f"🔍 Non-Member Query: {non_member_query}")
    # result = await Runner.run(library_agent, non_member_query, context=ctx_bob, config=run_config)
    result = await Runner.run(library_agent, non_member_query, context=ctx_bob)
    print("💬 Agent Response:", _extract_response_text(result), "\n")


# ------------- Sync test using Runner.run_sync -------------
def ensure_event_loop_for_run_sync():
    """
    Older versions of some SDKs call asyncio.get_event_loop().run_until_complete inside run_sync.
    Ensure there's a loop available on main thread (Python 3.11+).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def sync_tests():
    # Make sure a loop exists for run_sync internals
    ensure_event_loop_for_run_sync()

    print("\n=== Sync run using Runner.run_sync ===\n")
    carol = UserContext(name="Carol", member_id="555-xyz")
    ctx_carol = LibraryRunContext(carol)
    q = "Find Big Data Analytics and tell me if it's available"
    print("🔍 User Query (sync):", q)
    # result = Runner.run_sync(library_agent, q, context=ctx_carol, config=run_config)
    result = Runner.run_sync(library_agent, q, context=ctx_carol)
    print("💬 Agent Response (sync):", _extract_response_text(result), "\n")


# ------------- Main -------------
if __name__ == "__main__":
    # Run async tests first
    asyncio.run(async_tests())

    # Then run sync tests (safe, after asyncio.run ended)
    sync_tests()
