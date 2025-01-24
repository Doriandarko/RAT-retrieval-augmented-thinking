# 🧠 RAT (Retrieval Augmented Thinking)

> *Enhancing AI responses through structured reasoning and knowledge retrieval*

RAT is a powerful tool that improves AI responses by leveraging DeepSeek's reasoning capabilities to guide other models through a structured thinking process.

## 💡 Origin & Ideation

The idea for RAT emerged from an interesting discovery about DeepSeek-R1 API capabilities. By setting the final response token to 1 while retrieving the thinking process, it became possible to separate the reasoning stage from the final response generation. This insight led to the development of a two-stage approach that combines DeepSeek's exceptional reasoning abilities with various response models.

Link to my original concept in this [Twitter thread](https://x.com/skirano/status/1881922469411643413).

## How It Works

RAT employs a two-stage approach:
1. **Reasoning Stage** (DeepSeek): Generates detailed reasoning and analysis for each query
2. **Response Stage** (OpenRouter): Utilizes the reasoning context to provide informed, well-structured answers

This approach ensures more thoughtful, contextually aware, and reliable responses.

## 🎯 Features

- 🤖 **Model Selection**: Flexibility to choose from various OpenRouter models
- 🧠 **Reasoning Visibility**: Toggle visibility of the AI's thinking
- 🔄 **Context Awareness**: Maintains conversation context for more coherent interactions

## ⚙️ Requirements

• Python 3.11 or higher  
• A .env file containing:
  ```plaintext
  DEEPSEEK_API_KEY=your_deepseek_api_key_here
  OPENROUTER_API_KEY=your_openrouter_api_key_here
  optional
  ANTHROPIC_API_KEY=your_anthropic_api_key_here
  ```

## 🚀 Installation
Standalone installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Doriandarko/RAT-retrieval-augmented-thinking.git
   cd RAT-retrieval-augmented-thinking
   ```


2. Install as a local package:
   ```bash
   pip install -e .
   ```

This will install RAT as a command-line tool, allowing you to run it from anywhere by simply typing `rat`!

## 📖 Usage

1. Ensure your .env file is configured with:
   ```plaintext
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   optional
   ANTHROPIC_API_KEY=your_anthropic_api_key_here 
   ```

2. Run RAT from anywhere:
   ```bash
   rat
   ```

3. Available commands:
   - Enter your question to get a reasoned response
   - Use "model <name>" to switch OpenRouter models
   - Type "reasoning" to show/hide the thinking process
   - Type "quit" to exit



## 🚀 Versions
You can also run each script on its own:

### Standard Version (rat.py)
The default implementation using DeepSeek for reasoning and OpenRouter for responses.
Run it using:
```bash
uv run rat.py
```

### Claude-Specific Version (rat-claude.py)
A specialized implementation designed for Claude models that leverages Anthropic's message prefilling capabilities. This version makes Claude believe the reasoning process is its own internal thought process, leading to more coherent and contextually aware responses.
Run it using:
```bash
uv run rat-claude.py
```


## 🤝 Contributing

Interested in improving RAT?

1. Fork the repository
2. Create your feature branch
3. Make your improvements
4. Submit a Pull Request

## 📜 License

This project is available under the MIT License. See the [LICENSE](LICENSE) file for details.

If you use this codebase in your projects, please include appropriate credits:

```plaintext
This project uses RAT (Retrieval Augmented Thinking) by Skirano
GitHub: https://github.com/Doriandarko/RAT-retrieval-augmented-thinking
```
---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Doriandarko/RAT-retrieval-augmented-thinking&type=Date)](https://star-history.com/#Doriandarko/RAT-retrieval-augmented-thinking&Date)

