import os
import time
from openai import OpenAI

# --- Configuration ---
# For security, it's best practice to set your API key as an environment variable.
# In your terminal, you can run:
# export OPENROUTER_API_KEY='your-actual-openrouter-api-key'
#
# This script will then read the key from the environment.
OPENROUTER_API_KEY = os.get_env("OPENROUTER_API_KEY")
MODEL_ID = "deepseek/deepseek-prover-v2" # Use the model identifier from OpenRouter

# --- Verify API Key ---
if not OPENROUTER_API_KEY:
    raise ValueError(
        "OpenRouter API key not found. "
        "Please set the OPENROUTER_API_KEY environment variable."
    )

# --- Initialize API Client ---
# We configure the OpenAI client to point to OpenRouter's API endpoint.
try:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
except Exception as e:
    print(f"Error initializing the OpenAI client: {e}")
    exit()

# --- Define the Prompt and Formal Statement ---
# This is the original Lean 4 statement you provided.
formal_statement = """
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- What is the positive difference between $120\%$ of 30 and $130\%$ of 20? Show that it is 10.-/
theorem mathd_algebra_10 : abs ((120 : ℝ) / 100 * 30 - 130 / 100 * 20) = 10 := by
  sorry
""".strip()

# This is the instruction for the model.
prompt_template = """
Complete the following Lean 4 code:

```lean4
{}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()

# We format the final prompt with the formal statement.
final_prompt = prompt_template.format(formal_statement)

# --- Prepare the API Request ---
# The chat history is formatted as a list of dictionaries.
chat_payload = [
  {"role": "user", "content": final_prompt},
]

print(f"Sending request to model: {MODEL_ID}...")
print("-" * 30)

# --- Make the API Call ---
start_time = time.time()
try:
    # We call the chat completions endpoint with our payload.
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=chat_payload,
        max_tokens=8192,      # Corresponds to 'max_new_tokens'
        temperature=0,        # For deterministic and focused output
        top_p=1,              # Default, but good to be explicit
        stream=False,         # We will wait for the full response
    )

    # --- Process and Display the Response ---
    end_time = time.time()
    elapsed_time = end_time - start_time

    if response.choices:
        # Extract the content from the first choice in the response.
        model_output = response.choices[0].message.content
        print("--- Model Response ---")
        print(model_output)
    else:
        print("No response was received from the model.")

    print("-" * 30)
    print(f"Request completed in {elapsed_time:.2f} seconds.")

except Exception as e:
    print(f"An error occurred during the API call: {e}")


