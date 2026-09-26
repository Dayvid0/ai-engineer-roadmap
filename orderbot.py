import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

def get_completion_from_messages(messages, model="openai/gpt-oss-120b", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content

context = [{'role': 'system', 'content': """
You are OrderBot, an automated service to collect orders for a pizza restaurant.
You first greet the customer, then collect the order,
and then ask if it's a pickup or delivery.
You wait to collect the entire order, then summarize it and check one final
time if the customer wants to add anything else.
If it's a delivery, you ask for an address.
Finally you collect the payment.
Make sure to clarify all options, extras and sizes to uniquely
identify the item from the menu.
You respond in a short, very conversational, friendly style.
The menu includes
pepperoni pizza 12.95, 10.00, 7.00
cheese pizza 10.95, 9.25, 6.50
eggplant pizza 11.95, 9.75, 6.75
fries 4.50, 3.50
greek salad 7.25
Toppings:
extra cheese 2.00, mushrooms 1.50, sausage 3.00, canadian bacon 3.50, AI sauce 1.50, peppers 1.00
Drinks:
coke 3.00, 2.00, 1.00, sprite 3.00, 2.00, 1.00, bottled water 5.00
"""}]

print("OrderBot ready! Type 'quit' to stop and get your JSON order summary.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    context.append({'role': 'user', 'content': user_input})
    response = get_completion_from_messages(context)
    context.append({'role': 'assistant', 'content': response})
    print(f"OrderBot: {response}\n")

# After the conversation ends, generate a structured JSON summary
context.append({'role': 'system', 'content': """
Create a JSON summary of the previous food order. Itemize the price for each item.
The fields should be:
1) pizza, include size
2) list of toppings
3) list of drinks, include size
4) list of sides, include size
5) total price
"""})
summary = get_completion_from_messages(context, temperature=0)
print("\n--- ORDER SUMMARY (JSON) ---")
print(summary)