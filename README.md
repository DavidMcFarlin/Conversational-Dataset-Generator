# Synthetic Conversational Dataset Generator

A lightweight, multi-threaded Python pipeline designed to give you a headstart in creating synthetic, multi-turn conversational datasets for LLM fine-tuning.

Whether you are trying to build a custom AI persona, train a model on specific domain knowledge, or break a model out of its standard RLHF "assistant" tone, this repository provides the structural foundation to generate high-quality JSONL training data via API.

## Features

* **Configuration-Driven:** Zero need to edit Python code. Define your topics, goals, and API settings entirely in `config.yaml`.
* **Dedicated Persona Injection:** Use `persona.md` to define your character's exact tone, behavioral principles, and prohibited phrases using standard Markdown.
* **Multi-Threaded Generation:** Utilizes multiple concurrent workers to rapidly generate dataset rows.
* **Format-Enforced Output:** Prompts are structured to enforce alternating turns and strict dialogue limits, saving directly to `.jsonl` files ready for processing.
* **Provider Agnostic:** Built using the standard OpenAI client, meaning it can easily be pointed at OpenAI, MiniMax, TogetherAI, or local inference servers simply by changing the `base_url` and API key.

## Setup & Installation

**1. Clone the repository:**
```bash
git clone [https://github.com/DavidMcFarlin/Conversational-Dataset-Generator.git](https://github.com/DavidMcFarlin/Conversational-Dataset-Generator.git)
cd Conversational-Dataset-Generator
```

**2. Install dependencies:**
Make sure you have Python 3.8+ installed. Note the addition of `pyyaml`.
```bash
pip install openai python-dotenv pyyaml
```

**3. Configure Environment Variables:**
Create a `.env` file in the root directory and add your chosen API key and base URL (if overriding the default):
```ini
MINIMAX_API_KEY=your_api_key_here
# Optional overrides:
# API_BASE_URL=[https://api.openai.com/v1](https://api.openai.com/v1)
# MODEL=model_name
```

## ⚙️ Usage & Customization

This generator separates the logic from the data. Before running the script, customize the two configuration files to match your desired training target:

### 1. `config.yaml`
This is the central control hub. Open it to define:
* `user_name` and `assistant_name`
* `target_per_category` (How many conversations to generate per topic)
* `available_subjects` (The list of topics the generator will iterate through)
* `conversation_goals` and `tone_calibration` for each subject
* `generation_settings` (workers, max tokens, temperature)

### 2. `persona.md`
This is the "soul" of your dataset. Edit this markdown file to define:
* **Core Identity:** Who the assistant is and what they represent.
* **Behavioral Principles:** Rules for how they speak (e.g., "NEVER theatrical," "Stays grounded").
* **Voice Examples:** Few-shot examples of how the assistant should respond in different modes.
* **Prohibited Phrases:** A blacklist of "AI boilerplate" you want the model to actively unlearn (e.g., "As an AI...").

### 3. Run the Generator
Once your config and persona files are set, simply run:
```bash
python generator.py
```

The script will automatically create your defined output directory (default: `raw_conversations/`) and populate it with individual `.jsonl` files for each subject.

## 📄 License
Released under the Apache 2.0 License.
