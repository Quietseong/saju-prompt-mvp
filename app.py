import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")
import json
import gradio as gr
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import signal
import sys
import argparse  # Add for command line arguments parsing

from src.llm_client import LLMClient, ModelType
from src.prompt_manager import PromptManager
from src.models import BirthInfo, PersonaData
from src.utils import validate_birth_info

# Default settings - HuggingFace Llama3 model
DEFAULT_HF_MODEL = "meta-llama/Llama-3-8b-instruct"
DEFAULT_TEMPLATE = "summary_v1"  # Default template fixed

# Load sample persona data
def load_sample_personas() -> List[Dict[str, Any]]:
    sample_path = Path("data/sample_personas.json")
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# Load tone/style data
def load_tone_examples() -> Dict[str, Dict[str, Any]]:
    """Load tone/style examples"""
    tone_path = Path("data/tone_example.json")
    if tone_path.exists():
        try:
            with open(tone_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load tone examples: {str(e)}")
    return {}

# Get tone/style selection options based on tone_example.json
def get_tone_options() -> List[Tuple[str, str]]:
    """Get list of tone/style choices from tone_example.json with fallback options"""
    # First try to load from the JSON file
    options = [("default", "Default Style")]
    
    try:
        tones = load_tone_examples()
        if tones:
            for tone_key, tone_data in tones.items():
                description = tone_data.get("description", "")
                options.append((tone_key, f"{tone_key.capitalize()} - {description}"))
        else:
            # Fallback if no tones loaded
            options.extend([
                ("cute", "Cute - Friendly and adorable style"),
                ("serious", "Serious - Formal and composed style"),
                ("direct", "Direct - Straightforward, no-nonsense style"),
                ("friendly_local", "Friendly Local - Casual and supportive style")
            ])
    except Exception as e:
        print(f"Warning: Error generating tone options: {str(e)}")
        # Fallback if exception occurs
        options = [
            ("default", "Default Style"),
            ("cute", "Cute - Friendly and adorable style"),
            ("serious", "Serious - Formal and composed style"),
            ("direct", "Direct - Straightforward, no-nonsense style"),
            ("friendly_local", "Friendly Local - Casual and supportive style")
        ]
    
    return options

# Initialize prompt manager
prompt_manager = PromptManager(templates_dir="prompts/templates")

# Initialize client function - HuggingFace API only
def init_client(api_key: str = None) -> LLMClient:
    """Initialize LLM client (HuggingFace API only)"""
    api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise ValueError("HuggingFace API key is required. Please provide an api_key parameter or set the HUGGINGFACE_API_KEY environment variable.")
    return LLMClient(
        provider_type=ModelType.HUGGINGFACE,
        model=DEFAULT_HF_MODEL,
        api_key=api_key
    )

# Check if user info is complete
def is_user_info_set(user_info: Dict[str, Any]) -> bool:
    """Check if all user information is set"""
    required_fields = ["name", "birth_year", "birth_month", "birth_day", "birth_hour", "birth_minute", "gender"]
    return all(field in user_info and user_info[field] is not None for field in required_fields)

# Generate chatbot response
def generate_chat_response(
    message: str,
    chat_history: List[Dict[str, str]],
    user_info: Dict[str, Any],
    api_key: str,
    tone: str = "default"
) -> Tuple[List[Dict[str, str]], Dict[str, Any], Dict[str, Any]]:
    """
    Generate chatbot response to user message
    
    Args:
        message: User message
        chat_history: Previous conversation history as a list of message dictionaries
                     with 'role' ('user' or 'assistant') and 'content' keys
        user_info: User information dictionary
        api_key: HuggingFace API key
        tone: Chatbot tone/style
    
    Returns:
        Updated conversation history, updated user info, model info
    """
    # User info collection phase
    if not is_user_info_set(user_info):
        # Request needed information if incomplete
        if "step" not in user_info:
            user_info["step"] = "name"
            return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Hello! I need some information for your Saju reading. First, what's your name?"}], user_info, {}
            
        if user_info["step"] == "name":
            user_info["name"] = message
            user_info["step"] = "birth_year"
            return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Thank you. What year were you born? (e.g., 1990)"}], user_info, {}
            
        if user_info["step"] == "birth_year":
            try:
                user_info["birth_year"] = int(message)
                user_info["step"] = "birth_month"
                return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": "What month were you born? (1-12)"}], user_info, {}
            except ValueError:
                return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Please enter a number. What year were you born?"}], user_info, {}
                
        if user_info["step"] == "birth_month":
            try:
                month = int(message)
                if 1 <= month <= 12:
                    user_info["birth_month"] = month
                    user_info["step"] = "birth_day"
                    return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": "What day of the month were you born? (1-31)"}], user_info, {}
                else:
                    return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Please enter a value between 1 and 12."}], user_info, {}
            except ValueError:
                return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Please enter a number. What month were you born?"}], user_info, {}
                
        if user_info["step"] == "birth_day":
            try:
                day = int(message)
                if 1 <= day <= 31:  # Simple validation only (skipping month-specific day validation)
                    user_info["birth_day"] = day
                    user_info["step"] = "birth_hour"
                    return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": "What hour were you born? (0-23 hour format, enter 12 if unknown)"}], user_info, {}
                else:
                    return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Please enter a value between 1 and 31."}], user_info, {}
            except ValueError:
                return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Please enter a number. What day were you born?"}], user_info, {}
                
        if user_info["step"] == "birth_hour":
            try:
                hour = int(message)
                if 0 <= hour <= 23:
                    user_info["birth_hour"] = hour
                    user_info["step"] = "birth_minute"
                    return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": "What minute were you born? (0-59, enter 0 if unknown)"}], user_info, {}
                else:
                    return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Please enter a value between 0 and 23."}], user_info, {}
            except ValueError:
                return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Please enter a number. What hour were you born?"}], user_info, {}
                
        if user_info["step"] == "birth_minute":
            try:
                minute = int(message)
                if 0 <= minute <= 59:
                    user_info["birth_minute"] = minute
                    user_info["step"] = "gender"
                    return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": "What is your gender? ('male' or 'female')"}], user_info, {}
                else:
                    return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Please enter a value between 0 and 59."}], user_info, {}
            except ValueError:
                return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Please enter a number. What minute were you born?"}], user_info, {}
                
        if user_info["step"] == "gender":
            if message.lower() in ["male", "m", "man"]:
                user_info["gender"] = "male"
                user_info["step"] = "complete"
                resp = "Thank you for providing all the information. Please wait a moment while I prepare your Saju reading..."
                return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": resp}], user_info, {}
            elif message.lower() in ["female", "f", "woman"]:
                user_info["gender"] = "female"
                user_info["step"] = "complete"
                resp = "Thank you for providing all the information. Please wait a moment while I prepare your Saju reading..."
                return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": resp}], user_info, {}
            else:
                return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Please enter 'male' or 'female'."}], user_info, {}
    
    # Begin Saju reading once user info is complete
    try:
        # Verify API key is provided
        if not api_key:
            return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Error: HuggingFace API key is required. Please enter your API key in the settings panel."}], user_info, {}
        
        # Create BirthInfo object and validate
        birth_info = BirthInfo(
            year=user_info["birth_year"],
            month=user_info["birth_month"],
            day=user_info["birth_day"],
            hour=user_info["birth_hour"],
            minute=user_info["birth_minute"]
        )
        validate_birth_info(birth_info)
        
        # Create persona data
        persona = PersonaData(
            id="custom",
            name=user_info["name"],
            birth_info=birth_info,
            gender=user_info["gender"]
        )
        
        # Initialize client - HuggingFace API only
        client = init_client(api_key)
        
        # Include conversation history in prompt
        template_key = DEFAULT_TEMPLATE  # already English
        
        # Generate base prompt using template
        base_prompt = prompt_manager.generate_prompt(template_key, persona)
        
        # Apply tone/style from tone_example.json
        if tone and tone != "default":
            # Load tone data from file
            loaded_tones = load_tone_examples()
            
            # Check if the selected tone exists in loaded data
            if tone in loaded_tones:
                tone_data = loaded_tones[tone]
                tone_description = tone_data.get("description", "")
                tone_style = tone_data.get("tone", "")
                style_guidelines = tone_data.get("style_guidelines", {})
                
                # Add tone/style instruction
                tone_instruction = f"\n\nResponse Style: {tone_style}. {tone_description}\n"
                
                # Add style guidelines
                if style_guidelines:
                    tone_instruction += "Style Guidelines:\n"
                    for key, value in style_guidelines.items():
                        if key == "vocabulary" and isinstance(value, list):
                            tone_instruction += f"- Vocabulary to use: {', '.join(value)}\n"
                        elif key == "emoji_usage" and value is True:
                            tone_instruction += f"- Use emojis freely to express emotions\n"
                        elif key == "emoji_usage" and value is False:
                            tone_instruction += f"- Avoid using emojis\n"
                        elif key == "sentence_length":
                            tone_instruction += f"- Use {value} sentences\n"
                        elif key == "exclamation_preference":
                            tone_instruction += f"- Use exclamation marks {value}\n"
                        elif key == "metaphor_level":
                            tone_instruction += f"- Use {value} level of metaphorical language\n"
                        elif key == "formality":
                            tone_instruction += f"- Maintain {value} level of formality\n"
                        else:
                            tone_instruction += f"- {key}: {value}\n"
                
                base_prompt += tone_instruction
        
        # Add previous conversation history
        conversation_history = "\n\nConversation History:\n"
        for i, msg in enumerate(chat_history[-5:]):  # Include only the last 5 exchanges
            conversation_history += f"User: {msg['content'] if msg['role'] == 'user' else ''}\nChatbot: {msg['content'] if msg['role'] == 'assistant' else ''}\n"
        
        # Add current user message
        conversation_history += f"User: {message}\nChatbot: "
        
        # Final prompt
        full_prompt = base_prompt + conversation_history
        
        # Call LLM with fixed temperature
        response = client.generate_saju_reading(
            prompt=full_prompt,
            temperature=0.7,  # Fixed temperature
            max_tokens=2000
        )
        
        # Extract result
        result_text = response["text"]
        
        # Prepare model info
        model_info = {
            "model": response["model"],
            "latency": response.get("latency", 0),
            "tokens": response.get("usage", {})
        }
        
        # Update conversation history
        updated_history = chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": result_text}]
        
        return updated_history, user_info, model_info
        
    except Exception as e:
        # Handle errors
        error_message = f"An error occurred: {str(e)}"
        return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": error_message}], user_info, {}

# Create Gradio chatbot interface
def create_interface():
    # Title and description
    title = "🔮 SajuMate - Eastern Fortune Reading Chatbot"
    description = """
    ## 🔮 SajuMate - Eastern Fortune Reading Chatbot
    
    Chat with our AI to receive a personalized Eastern fortune reading based on your birth information.
    The chatbot will guide you through the process of providing your name, birth date/time, and gender.
    """
    
    # UI layout
    with gr.Blocks(title=title) as demo:
        gr.Markdown(description)
        
        # State variables
        user_info = gr.State({})  # Store user information
        model_info_state = gr.State({})  # Store model information
        
        with gr.Row():
            with gr.Column(scale=3):
                # Chatbot UI - Fix type parameter
                chatbot = gr.Chatbot(
                    type="messages",  # Ensure using messages type instead of tuples
                    label="Saju Reading Conversation",
                    height=500
                )
                
                # Message input
                with gr.Row():
                    message = gr.Textbox(
                        placeholder="Type your message here...",
                        label="Message",
                        show_label=False
                    )
                    submit_btn = gr.Button("Send", variant="primary")
                
                # Clear conversation button
                clear_btn = gr.Button("Reset Conversation")
                
            with gr.Column(scale=1):
                # Chatbot tone/style selection - Highlighted with a container
                with gr.Group():
                    gr.Markdown("### Chatbot Personality")
                    tone_style = gr.Dropdown(
                        choices=get_tone_options(),
                        value="default",
                        label="Select a tone/style for the chatbot"
                    )
                
                # API key input with better guidance
                with gr.Group():
                    gr.Markdown("### API Settings")
                    api_key = gr.Textbox(
                        label="HuggingFace API Key", 
                        placeholder="Enter your HuggingFace API Key",
                        value=os.getenv("HUGGINGFACE_API_KEY", ""),
                        type="password",
                        info="Make sure you have access to Llama-3 models on HuggingFace"
                    )
                    api_info = gr.Markdown("""
                    **Note:** You must have access to Meta's Llama-3 models on HuggingFace.
                    1. Sign up/login at [HuggingFace](https://huggingface.co/)
                    2. Request access to [Llama-3-8b-instruct](https://huggingface.co/meta-llama/Llama-3-8b-instruct)
                    3. Generate API key in your [account settings](https://huggingface.co/settings/tokens)
                    """)
                
                # Model info display
                with gr.Group():
                    gr.Markdown("### Session Info")
                    model_info_text = gr.Textbox(label="Model Information", lines=2)
        
        # Update model info function
        def update_model_info(model_info_state):
            if not model_info_state:
                return ""
                
            info_text = f"Model: {model_info_state.get('model', 'N/A')}"
            
            # Add token info
            tokens = model_info_state.get("tokens", {})
            if tokens:
                info_text += f"\nTokens: {tokens.get('total_tokens', 'N/A')} (input: {tokens.get('prompt_tokens', 'N/A')}, output: {tokens.get('completion_tokens', 'N/A')})"
            
            # Add latency info
            latency = model_info_state.get("latency")
            if latency:
                info_text += f"\nResponse time: {latency:.2f} seconds"
                
            return info_text
        
        # Handle message submission with loading indicator
        def on_message(message, chat_history, user_info, model_info_state, progress=gr.Progress()):
            if message.strip() == "":
                return message, chat_history, user_info, model_info_state, ""
            
            # Add loading steps
            progress(0, desc="Processing message...")
            
            # Update UI to show user message immediately
            interim_history = chat_history + [{"role": "user", "content": message}]
            
            progress(0.3, desc="Generating response...")
            updated_history, updated_user_info, new_model_info = generate_chat_response(
                message=message,
                chat_history=chat_history,
                user_info=user_info,
                api_key=api_key.value,
                tone=tone_style.value
            )
            
            progress(0.8, desc="Finalizing...")
            info_text = update_model_info(new_model_info)
            
            progress(1.0, desc="Done!")
            return "", updated_history, updated_user_info, new_model_info, info_text
        
        # Reset conversation function
        def clear_chat():
            return [], {}, {}, ""
        
        # Connect events
        submit_btn.click(
            fn=on_message,
            inputs=[message, chatbot, user_info, model_info_state],
            outputs=[message, chatbot, user_info, model_info_state, model_info_text],
        )
        
        message.submit(
            fn=on_message,
            inputs=[message, chatbot, user_info, model_info_state],
            outputs=[message, chatbot, user_info, model_info_state, model_info_text],
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, user_info, model_info_state, model_info_text]
        )
        
        model_info_state.change(
            fn=update_model_info,
            inputs=[model_info_state],
            outputs=[model_info_text]
        )
    
    return demo

# Main application entry point
def main(share=False):
    """Console script entry point"""
    # Enable better error handling for API key validation
    try:
        demo = create_interface()
        # Run with default settings
        port = int(os.getenv("SAJUMATE_PORT", "7861"))  # Changed default port to avoid conflicts
        server_name = os.getenv("SAJUMATE_HOST", "0.0.0.0")
        
        print(f"Starting SajuMate Chatbot UI: http://{server_name if server_name != '0.0.0.0' else 'localhost'}:{port}")
        print(f"Using model: HuggingFace API - {DEFAULT_HF_MODEL}")
        if share:
            print("Public URL sharing is enabled - perfect for Colab!")
        else:
            print("To run with Colab, add --share to enable public URL sharing")
        
        # Add graceful error handling for launch
        try:
            demo.launch(
                server_name=server_name,
                server_port=port,
                share=share
            )
        except Exception as e:
            print(f"Error launching Gradio interface: {str(e)}")
            print("If running in Colab, make sure to use share=True")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Set up a proper signal handler for graceful exit
    def signal_handler(sig, frame):
        print("Shutting down gracefully...")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SajuMate - Eastern Fortune Reading Chatbot')
    parser.add_argument('--share', action='store_true', help='Enable public URL sharing (required for Colab)')
    args = parser.parse_args()
    
    # Run with share=True if specified or in Colab environment
    is_colab = 'google.colab' in sys.modules
    if is_colab:
        print("Colab environment detected, enabling public URL sharing")
        share = True
    else:
        share = args.share
        
    main(share=share) 