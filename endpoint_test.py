from openai import AzureOpenAI, AsyncAzureOpenAI
import asyncio

# Create a client for Azure OpenAI
chat_client = AzureOpenAI(
    api_key="5UnXrfATc5KyXEVyxpjeJF9MInBazuoBMBkyHKEB1nARFKxuJGLtJQQJ99BCAC4f1cMXJ3w3AAABACOGNEeL",
    api_version="2024-08-01-preview",
    azure_endpoint="https://interviews3.openai.azure.com/"
)

# Also create async client
async_client = AsyncAzureOpenAI(
    api_key="5UnXrfATc5KyXEVyxpjeJF9MInBazuoBMBkyHKEB1nARFKxuJGLtJQQJ99BCAC4f1cMXJ3w3AAABACOGNEeL",
    api_version="2024-08-01-preview",
    azure_endpoint="https://interviews3.openai.azure.com/"
)

# --- Test Synchronous Client --- (Existing Code)
print("--- Testing Synchronous Client ---")
try:
    response = chat_client.chat.completions.create(
        model="gpt-35-turbo",  # This is your Azure *deployment name*, not the model name
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a fun fact about space (sync)."}
        ]
    )
    # Print the response
    print("Sync Response:", response.choices[0].message.content)
except Exception as e:
    print(f"Error testing synchronous client: {e}")

# --- Test Asynchronous Client --- (New Code)
print("\n--- Testing Asynchronous Client ---")

async def test_async():
    try:
        response = await async_client.chat.completions.create(
            model="gpt-35-turbo",  # Deployment name
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a fun fact about space (async)."}
            ]
        )
        print("Async Response:", response.choices[0].message.content)
    except Exception as e:
        print(f"Error testing asynchronous client: {e}")

# Run the async test function
if __name__ == "__main__":
    # Keep the sync test running first
    # Then run the async test
    asyncio.run(test_async())

