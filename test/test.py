import os
import json
import boto3
from dotenv import load_dotenv

load_dotenv()

region = os.getenv("AWS_REGION")
access_key_id = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY")
secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
embed_model = os.getenv("BEDROCK_EMBED_MODEL")
chat_model = os.getenv("BEDROCK_CHAT_MODEL")

missing = [
    name
    for name, value in {
        "AWS_REGION": region,
        "AWS_ACCESS_KEY_ID or AWS_ACCESS_KEY": access_key_id,
        "AWS_SECRET_ACCESS_KEY": secret_access_key,
        "BEDROCK_EMBED_MODEL": embed_model,
        "BEDROCK_CHAT_MODEL": chat_model,
    }.items()
    if not value
]
if missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

client = boto3.client(
    "bedrock-runtime",
    region_name=region,
    aws_access_key_id=access_key_id,
    aws_secret_access_key=secret_access_key,
)

# Test 1: Amazon Titan Embeddings
embedding_response = client.invoke_model(
    modelId=embed_model,
    body=json.dumps({
        "inputText": "Hello Amazon Bedrock",
        "dimensions": 1024,
        "normalize": True
    }),
    contentType="application/json",
    accept="application/json"
)

embedding_data = json.loads(
    embedding_response["body"].read()
)

print("Embedding dimension:", len(embedding_data["embedding"]))

# Test 2: Amazon Nova Lite
chat_response = client.converse(
    modelId=chat_model,
    messages=[
        {
            "role": "user",
            "content": [
                {"text": "Explain RAG in one sentence."}
            ]
        }
    ],
    inferenceConfig={"maxTokens": 200}
)

print(
    "Nova Response:",
    chat_response["output"]["message"]["content"][0]["text"]
)