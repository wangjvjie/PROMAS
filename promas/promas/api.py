
from openai import OpenAI

def openai_send_messages(message, temperature=0.8, n_repeat=1):
    model = 'deepseek-chat'
    client = OpenAI(base_url='https://api.deepseek.com', api_key='YOUR_API_KEY')
        
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message}],
        temperature=0.4,
        max_completion_tokens=22000,
        n=n_repeat,
    )
    
    return response.choices[0].message.content