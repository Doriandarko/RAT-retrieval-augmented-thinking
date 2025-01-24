from openai import OpenAI
import os
import re
from dotenv import load_dotenv
from rich import print as rprint
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
import time


# Model Constants
DEEPSEEK_MODEL = "deepseek-r1-distill-llama-8b"
RESPONSE_MODEL = "meta-llama-3.1-8b-instruct"

# Don't forget to add LOCAL_API_KEY to .env file.

# Load environment variables
load_dotenv()

class ModelChain:
    def __init__(self):
        # Initialize Local DeepSeek client
        self.deepseek_client = OpenAI(
            api_key=os.getenv("LOCAL_API_KEY"),
            base_url="http://localhost:1234/v1"
        )

        # Initialize response_client
        self.response_client = OpenAI(
            api_key=os.getenv("LOCAL_API_KEY"),
            base_url="http://localhost:1234/v1"
        )

        self.deepseek_messages = []
        self.response_messages = []
        self.current_model = RESPONSE_MODEL
        self.show_reasoning = True

    def set_model(self, model_name):
        self.current_model = model_name

    def get_model_display_name(self):
        return self.current_model

    def get_deepseek_reasoning(self, user_input):
        start_time = time.time()
        self.deepseek_messages.append({"role": "user", "content": user_input})

        if self.show_reasoning:
            rprint("\n[blue]Reasoning Process[/]")

        response = self.deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=self.deepseek_messages,
            stream=True
        )

        full_response = ""
        is_thinking = False
        think_content = ""

        # Concatenate streamed chunks to form the full response
        for chunk in response:
            if hasattr(chunk, "choices") and chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    if "<think>" in content:
                        is_thinking = True
                        think_content = ""
                    if is_thinking:
                        think_content += content
                        if "</think>" in content:
                            is_thinking = False
        extracted_think_content = re.findall(r'<think>(.*?)</think>', think_content, re.DOTALL)
        print("\n".join(extracted_think_content))

        # Extract content inside <think> tags
        elapsed_time = time.time() - start_time
        if elapsed_time >= 60:
            time_str = f"{elapsed_time/60:.1f} minutes"
        else:
            time_str = f"{elapsed_time:.1f} seconds"
        rprint(f"\n\n[yellow]Thought for {time_str}[/]")

        if self.show_reasoning:
            print("\n")
        return extracted_think_content

    def get_local_response(self, user_input, reasoning):
        # Create messages with proper format
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_input
                }
            ]
        }

        assistant_prefill = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"<thinking>{reasoning}</thinking>"
                }
            ]
        }

        messages = [user_message, assistant_prefill]

        rprint(f"[green]{self.get_model_display_name()}[/]", end="")

        try:
            response = self.response_client.chat.completions.create(
            model=self.current_model,
            messages=messages,
            max_tokens=8000,
            stream=True  # Make sure stream is set to True
            )

            full_response = ""
            # Iterate over the streamed response directly or text in response:
            for text in response:
                if text.choices[0].delta.content is not None:
                    full_response += text.choices[0].delta.content
                    print(text.choices[0].delta.content, end="", flush=True)

            self.response_messages.extend([
                user_message,
                {
                    "role": "assistant", 
                    "content": [{"type": "text", "text": full_response}]
                }
            ])
            self.deepseek_messages.append({"role": "assistant", "content": full_response})

            print("\n")
            return full_response

        except Exception as e:
            rprint(f"\n[red]Error in response: {str(e)}[/]")
            return "Error occurred while getting response"

def main():
    chain = ModelChain()

    style = Style.from_dict({
        'prompt': 'orange bold',
    })
    session = PromptSession(style=style)

    rprint(Panel.fit(
        "[bold cyan]Retrival augmented thinking[/]",
        title="[bold cyan]Local RAT 🧠[/]",
        border_style="cyan"
    ))
    rprint("[yellow]Commands:[/]")
    rprint(" • Type [bold red]'quit'[/] to exit")
    rprint(" • Type [bold magenta]'model <name>'[/] to change the Response model")
    rprint(" • Type [bold magenta]'reasoning'[/] to toggle reasoning visibility")
    rprint(" • Type [bold magenta]'clear'[/] to clear chat history\n")
    
    while True:
        try:
            user_input = session.prompt("\nYou: ", style=style).strip()

            if user_input.lower() == 'quit':
                print("\nGoodbye! 👋")
                break

            if user_input.lower() == 'clear':
                chain.deepseek_messages = []
                chain.claude_messages = []
                rprint("\n[magenta]Chat history cleared![/]\n")
                continue

            if user_input.lower().startswith('model '):
                new_model = user_input[6:].strip()
                chain.set_model(new_model)
                print(f"\nChanged model to: {chain.get_model_display_name()}\n")
                continue

            if user_input.lower() == 'reasoning':
                chain.show_reasoning = not chain.show_reasoning
                status = "visible" if chain.show_reasoning else "hidden"
                rprint(f"\n[magenta]Reasoning process is now {status}[/]\n")
                continue

            reasoning = chain.get_deepseek_reasoning(user_input)
            local_response = chain.get_local_response(user_input, reasoning)

        except KeyboardInterrupt:
            continue
        except EOFError:
            break

if __name__ == "__main__":
    main()