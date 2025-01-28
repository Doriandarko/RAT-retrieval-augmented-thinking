from openai import OpenAI
import os
from dotenv import load_dotenv
from rich import print as rprint
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
import time
import argparse
import json

# Model Constants
DEEPSEEK_MODEL = "DeepSeek-R1"
LLAMA_MODEL = "Meta-Llama-3-3-70B-Instruct"

# Load environment variables
load_dotenv()

class AkashModelChain:
    """
    A class that implements the RAT (Retrieval Augmented Thinking) interface on the Akash network.
    
    This class handles the communication with DeepSeek and Llama models, managing message history,
    and coordinating the reasoning and response generation process.
    
    Attributes:
        stream (bool): Whether to stream responses from the models
        client (OpenAI): The Akash API client
        deepseek_messages (list): History of messages for the DeepSeek model
        llama_messages (list): History of messages for the Llama model
        current_model (str): The currently selected Llama model
        show_reasoning (bool): Whether to display the reasoning process
    """
    
    def __init__(self, stream=True):
        """
        Initialize the ModelChain with specified streaming preference.
        
        Args:
            stream (bool, optional): Whether to stream model responses. Defaults to True.
        """
        self.stream = stream
        
        # Initialize Akash client
        self.client = OpenAI(
            api_key=os.getenv("AKASH_API_KEY"),
            base_url="https://chatapi.akash.network/api/v1"
        )

        self.deepseek_messages = []
        self.llama_messages = []
        self.current_model = LLAMA_MODEL
        self.show_reasoning = True

    def set_model(self, model_name):
        """
        Set the current model to use for responses.
        
        Args:
            model_name (str): The name/ID of the model to use
        """
        self.current_model = model_name

    def get_model_display_name(self):
        """
        Get the name of the currently selected model.
        
        Returns:
            str: The current model's name/ID
        """
        return self.current_model

    def get_all_models(self):
        """
        Retrieve a list of available models from the Akash client.
        
        Excludes DeepSeek-R1 models as they are primarily reasoning models 
        and not designed for generating direct responses. R1 models are 
        optimized for thinking/reasoning processes and are filtered out 
        to ensure only response-generation models are returned.
        
        Returns:
            List[str]: A list of model IDs available for response generation
        """
        models = self.client.models.list()
        return [model.id for model in models.data if not 'DeepSeek-R1' in model.id]

    def get_deepseek_reasoning(self, user_input):
        """
        Generate reasoning about the user's input using the DeepSeek model.
        
        This method sends the user's input to the DeepSeek model to generate
        a thought process or reasoning about how to respond to the query.
        
        Args:
            user_input (str): The user's input text
            
        Returns:
            str: The generated reasoning content, cleaned of <think> and </think> tags
        """
        start_time = time.time()
        self.deepseek_messages.append({"role": "user", "content": user_input})

        if self.show_reasoning:
            rprint("\n[blue]Reasoning Process[/]")

        response = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=self.deepseek_messages,
            stream=self.stream
        )

        reasoning_content = ""

        if self.stream:
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content_piece = chunk.choices[0].delta.content
                    reasoning_content += content_piece
                    if self.show_reasoning:
                        print(content_piece, end="", flush=True)
                    if "</think>" in content_piece:
                        break
        else:
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

    def get_llama_response(self, user_input, reasoning):
        """
        Generate a response using the selected model.
        
        This method takes both the user's input and the generated reasoning
        to produce a final response using the currently selected model.
        
        Args:
            user_input (str): The user's input text
            reasoning (str): The reasoning generated by DeepSeek
            
        Returns:
            str: The generated response from the model
        """
        # Create messages with reasoning included
        messages = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": f"<thinking>{reasoning}</thinking>"}
        ]

        rprint(f"[green]{self.get_model_display_name()}[/]", end="")

        try:
            if self.stream:
                full_response = ""
                response = self.client.chat.completions.create(
                    model=self.current_model,
                    messages=messages,
                    stream=self.stream
                )
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        print(text, end="", flush=True)
                        full_response += text
            else:
                response = self.client.chat.completions.create(
                    model=self.current_model,
                    messages=messages,
                    stream=self.stream
                )
                full_response = response.choices[0].message.content
                print(full_response)

            self.llama_messages.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": full_response}
            ])
            self.deepseek_messages.append({"role": "assistant", "content": full_response})

            print("\n")
            return full_response

        except Exception as e:
            rprint(f"\n[red]Error in response: {str(e)}[/]")
            return "Error occurred while getting response"

    def _save_message_history(self):
        """Save both message histories to JSON files"""
        try:
            with open('deepseek_messages.json', 'w', encoding='utf-8') as f:
                json.dump(self.deepseek_messages, f, indent=2, ensure_ascii=False)
                
            with open('llama_messages.json', 'w', encoding='utf-8') as f:
                json.dump(self.llama_messages, f, indent=2, ensure_ascii=False)
        except Exception as e:
            rprint(f"\n[red]Error saving message history: {str(e)}[/]")

def main():
    """
    Main function that runs the RAT Chat interface for the Akash network.
    
    This function sets up the command-line interface, processes user commands,
    and manages the interaction between the user and the model chain.
    """
    parser = argparse.ArgumentParser(description='RAT Chat Interface (Akash)')
    parser.add_argument('--no-stream', action='store_true',
                      help='Disable streaming responses')
    args = parser.parse_args()

    chain = AkashModelChain(stream=not args.no_stream)

    style = Style.from_dict({
        'prompt': 'orange bold',
    })
    session = PromptSession(style=style)

    rprint(Panel.fit(
        "[bold cyan]Retrieval Augmented Thinking - Akash Edition[/]",
        title="[bold cyan]RAT 🧠[/]",
        border_style="cyan"
    ))
    
    # Add streaming info to startup message
    stream_info = "disabled" if args.no_stream else "enabled"
    rprint(f"[yellow]Streaming: [bold]{stream_info}[/][/]")
    
    rprint("[yellow]Commands:[/]")
    rprint(" • Type [bold red]'quit'[/] to exit")
    rprint(" • Type [bold magenta]'model <name>'[/] to change the Llama model")
    rprint(" • Type [bold magenta]'reasoning'[/] to toggle reasoning visibility")
    rprint(" • Type [bold magenta]'view_models'[/] to see all available models")
    rprint(" • Type [bold magenta]'clear'[/] to clear chat history\n")
    
    while True:
        try:
            user_input = session.prompt("\nYou: ", style=style).strip()

            if user_input.lower() == 'quit':
                print("\nGoodbye! 👋")
                break

            if user_input.lower() == 'view_models':
                available_models = chain.get_all_models()
                rprint(f"\n[yellow]Available models:[/]")
                for model in available_models:
                    rprint(f" • {model}")
                print()
                continue

            if user_input.lower() == 'clear':
                chain.deepseek_messages = []
                chain.llama_messages = []
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
            llama_response = chain.get_llama_response(user_input, reasoning)

        except KeyboardInterrupt:
            continue
        except EOFError:
            break

if __name__ == "__main__":
    main() 