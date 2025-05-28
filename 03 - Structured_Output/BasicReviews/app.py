from langchain_ollama import ChatOllama
import streamlit as st
from typing import Annotated,Optional,Literal
from pydantic import BaseModel,Field

model = ChatOllama(model="deepseek-r1:7b")

class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key_themes in a list")
    summary: str = Field(description="A brief summary of review")
    sentiment:Literal["pos","neg"] = Field(description="Return sentiment is it Postive or Negative or Neutral")
    pros: Optional[list[str]] = Field(default=None,description="Write down all the pros inside a list if present in the review") 
    cons: Optional[list[str]] = Field(default=None,description="Write down all the cons inside a list if present in the review")
   
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
                          "I had high expectations for the Logitech MX Master 3S, but after using it for a few weeks, I found myself disappointed. While the ergonomic design is comfortable for long hours, the bulky shape might not be ideal for smaller hands. The silent clicks are a nice touch, making it great for office environments. However, the connectivity is unreliable, with frequent Bluetooth dropouts that force me to switch to the USB receiver. The MagSpeed scroll wheel, while impressive in theory, often feels inconsistent—sometimes too fast, other times lagging unpredictably. For a productivity-focused mouse, I expected better precision. The multi-device switching feature works well, but the overall experience is hindered by the connection issues. At nearly $100, I was hoping for a flawless experience, but the performance feels underwhelming for the price. If it were more reliable, it would be a great investment, but as it stands, it’s hard to justify the cost when other mice offer similar features without the frustration."
                          """)
review_result = result.model_dump()
print(review_result)
print()
print(f"Key Themes: {review_result['key_themes']}")
print()
print(f"Sentiment: {review_result['sentiment']}")
print()
print(f"Summary: {review_result['summary']}")
print()
print(f"Pros: {review_result['pros']}")
print()
print(f"Cons: {review_result['cons']}")