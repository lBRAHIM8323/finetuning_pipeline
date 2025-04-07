import gradio as gr
from openai import OpenAI
import json
from huggingface_hub import hf_hub_download

# Hardcoded API key and base URL
API_KEY = "NUKS7FDSZL0Q83OH1JDZ4U4C55PHXTAVY7C97YMI"
BASE_URL = "https://api.runpod.ai/v2/vllm-4eirfustzdfesg/openai/v1"

def generate_response(prompt, temperature, max_tokens, selected_model, model_list):
    # Find the selected model's details
    model_details = next((model for model in model_list if model['model'] == selected_model), None)

    if not model_details:
        return "Error: Model not found"

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    model_param = {
        "name": "assistant",
        "path": selected_model,
        "base_model_name": model_details['base_model']
    }

    try:
        # Create a chat completion
        response = client.chat.completions.create(
            model=json.dumps(model_param),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )

        # Return the response content
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def get_model_list():
    try:
        model_list_file = "model_list.json"

        # Download the model list file
        content_path = hf_hub_download(
            repo_id="PharynxAI/Model_List",
            filename=model_list_file,
            token="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",
            repo_type="dataset"
        )

        with open(content_path, "r") as file:
            model_list = json.load(file)

        return model_list
    except Exception as e:
        print(f"Error fetching model list: {e}")
        return []

def refresh_models():
    """Refresh the model list and return both model dropdown and details"""
    global model_list
    model_list = get_model_list()

    # Return both dropdown choices and the full model list
    return (
        gr.Dropdown(choices=[model["model"] for model in model_list],
                    value=model_list[0]["model"] if model_list else None),
        model_list  # Return the full model list for the JSON store
    )

# Create Gradio interface
with gr.Blocks(title="Finetune Interface Bot") as demo:
    gr.Markdown("""
        <div style="text-align: center;">
            <h1>Finetune Interface Bot🤖</h1>
        </div>
    """)
    # Fetch model list at startup
    model_list = get_model_list()

    # Dropdown for model selection
    model_name = gr.Dropdown(
        label="Select Model",
        choices=[model['model'] for model in model_list],
        value=[model['model'] for model in model_list][0] if model_list else None
    )

    # Hidden fields to store full model details
    model_details_store = gr.JSON(visible=False, value=model_list)

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                lines=5
            )
            with gr.Row():
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.01
                )
                max_tokens = gr.Slider(
                    label="Max Tokens",
                    minimum=100,
                    maximum=7000,
                    value=2000,
                    step=1
                )
            refresh_btn = gr.Button("🔄 Refresh Models")
            submit_btn = gr.Button("Generate Response",variant="primary")

    with gr.Row():
            output = gr.Textbox(label="Generated Response", lines=10)

    # Set up event handlers
    refresh_btn.click(
        fn=refresh_models,
        outputs=[model_name, model_details_store]
    )

    submit_btn.click(
        fn=generate_response,
        inputs=[prompt, temperature, max_tokens, model_name, model_details_store],
        outputs=output
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)