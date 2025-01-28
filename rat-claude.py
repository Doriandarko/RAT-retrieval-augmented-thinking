from openai import OpenAI
import os
import anthropic
from dotenv import load_dotenv
from rich import print as rprint
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
import time
import argparse
import json

# Model Constants
DEEPSEEK_MODEL = "deepseek-reasoner"
DEEPSEEK_MODEL_AKASH = "DeepSeek-R1"
CLAUDE_MODEL = "claude-3-5-haiku-20241022"

# Load environment variables
load_dotenv()

class ModelChain:
    def __init__(self, use_deepseek_openrouter=False, stream=True):
        self.use_deepseek_openrouter = use_deepseek_openrouter
        self.stream = stream
        
        # Initialize appropriate DeepSeek client based on use_deepseek flag
        if self.use_deepseek_openrouter:
            self.deepseek_client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )
            self.deepseek_model = DEEPSEEK_MODEL
        else:
            self.deepseek_client = OpenAI(
                api_key=os.getenv("AKASH_API_KEY"),
                base_url="https://chatapi.akash.network/api/v1"
            )
            self.deepseek_model = DEEPSEEK_MODEL_AKASH

        # Initialize Claude client
        self.claude_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        self.deepseek_messages = []
        self.claude_messages = []
        self.current_model = CLAUDE_MODEL
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

        # Create completion parameters
        completion_params = {
            "model": self.deepseek_model,
            "messages": self.deepseek_messages,
            "stream": self.stream
        }
        
        # Only add max_tokens if using OpenRouter (not Akash)
        if self.use_deepseek_openrouter:
            completion_params["max_tokens"] = 1

        response = self.deepseek_client.chat.completions.create(**completion_params)

        reasoning_content = ""
        final_content = ""

        if self.stream:
            for chunk in response:
                if self.use_deepseek_openrouter and chunk.choices[0].delta.reasoning_content:
                    # Original DeepSeek API handling
                    reasoning_piece = chunk.choices[0].delta.reasoning_content
                    reasoning_content += reasoning_piece
                    if self.show_reasoning:
                        print(reasoning_piece, end="", flush=True)
                    if "</think>" in reasoning_piece:
                        break
                elif chunk.choices[0].delta.content:
                    # Akash API or regular content handling
                    content_piece = chunk.choices[0].delta.content
                    if not self.use_deepseek_openrouter:
                        reasoning_content += content_piece
                        if self.show_reasoning:
                            print(content_piece, end="", flush=True)
                        if "</think>" in content_piece:
                            break
                    else:
                        final_content += content_piece
                        if "</think>" in final_content:
                            break
        else:
            # Non-streaming response handling
            content = response.choices[0].message.content
            if self.show_reasoning:
                print(content)
            reasoning_content = content
            if "</think>" in content:
                reasoning_content = content.split("</think>")[0] + "</think>"

        elapsed_time = time.time() - start_time
        if elapsed_time >= 60:
            time_str = f"{elapsed_time/60:.1f} minutes"
        else:
            time_str = f"{elapsed_time:.1f} seconds"
        rprint(f"\n\n[yellow]Thought for {time_str}[/]")

        if self.show_reasoning:
            print("\n")
        
        reasoning_content = reasoning_content.replace("<think>", "").replace("</think>", "")
        return reasoning_content

    def get_claude_response(self, user_input, reasoning):
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
            if self.stream:
                with self.claude_client.messages.stream(
                    model=self.current_model,
                    messages=messages,
                    max_tokens=8000
                ) as stream:
                    full_response = ""
                    for text in stream.text_stream:
                        print(text, end="", flush=True)
                        full_response += text
            else:
                response = self.claude_client.messages.create(
                    model=self.current_model,
                    messages=messages,
                    max_tokens=8000
                )
                full_response = response.content[0].text
                print(full_response)

            self.claude_messages.extend([
                user_message,
                {
                    "role": "assistant", 
                    "content": [{"type": "text", "text": full_response}]
                }
            ])
            self.deepseek_messages.append({"role": "assistant", "content": full_response})

            print("\n")
            
            # Save message histories to JSON files after each response
            # self._save_message_history()
            
            return full_response

        except Exception as e:
            rprint(f"\n[red]Error in response: {str(e)}[/]")
            return "Error occurred while getting response"
            
    def _save_message_history(self):
        """Save both message histories to JSON files"""
        try:
            with open('deepseek_messages.json', 'w', encoding='utf-8') as f:
                json.dump(self.deepseek_messages, f, indent=2, ensure_ascii=False)
                
            with open('claude_messages.json', 'w', encoding='utf-8') as f:
                json.dump(self.claude_messages, f, indent=2, ensure_ascii=False)
        except Exception as e:
            rprint(f"\n[red]Error saving message history: {str(e)}[/]")

def main():
    # Add argument parsing for API choice
    parser = argparse.ArgumentParser(description='RAT Chat Interface')
    parser.add_argument('--use-deepseek', action='store_true', 
                      help='Use original DeepSeek API instead of Akash Chat API')
    parser.add_argument('--no-stream', action='store_true',
                      help='Disable streaming responses')
    args = parser.parse_args()

    chain = ModelChain(
        use_deepseek_openrouter=args.use_deepseek,
        stream=not args.no_stream
    )

    style = Style.from_dict({
        'prompt': 'orange bold',
    })
    session = PromptSession(style=style)

    rprint(Panel.fit(
        "[bold cyan]Retrival augmented thinking[/]",
        title="[bold cyan]RAT 🧠[/]",
        border_style="cyan"
    ))
    
    # Add API info to startup message
    api_info = "Original DeepSeek API" if args.use_deepseek else "Akash Chat API"
    rprint(f"[yellow]Using: [bold]{api_info}[/][/]")
    
    # Add streaming info to startup message
    stream_info = "disabled" if args.no_stream else "enabled"
    rprint(f"[yellow]Streaming: [bold]{stream_info}[/][/]")
    
    rprint("[yellow]Commands:[/]")
    rprint(" • Type [bold red]'quit'[/] to exit")
    rprint(" • Type [bold magenta]'model <name>'[/] to change the Claude model")
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
            claude_response = chain.get_claude_response(user_input, reasoning)

        except KeyboardInterrupt:
            continue
        except EOFError:
            break

if __name__ == "__main__":
    main()