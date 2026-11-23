
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

comment = "Alhamdulillah semoga dipermudahkan urusan awak"

response = client.responses.create(
    model="gpt-4o-mini",
    instructions=(
        "You are a politeness classifier for Malaysian TikTok comments. "
        "Classify the comment into exactly one label: Polite or Impolite. "
        "Consider Malay, English, mixed language, slang and informal expressions. "
        "Return only the label: Polite or Impolite."
    ),
    input=comment
)

print("Comment:", comment)
print("LLM Prediction:", response.output_text.strip())

