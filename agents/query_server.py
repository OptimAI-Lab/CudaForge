from agents.llm_local import get_llm, GenerationConfig
import os

from utils.print_utils import print_bold

TOGETHER_KEY = os.environ.get("TOGETHER_API_KEY")
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
SGLANG_KEY = os.environ.get("SGLANG_API_KEY")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY")
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
from typing import Optional


def colorize_finish_reason(reason: Optional[str]) -> str:
    colors = {
        "stop": "\033[92m",  # Green
        "end_turn": "\033[92m",
        "length": "\033[93m",  # Yellow
        "max_tokens": "\033[93m",
        "content_filter": "\033[91m",  # Red
        "stop_sequence": "\033[91m",
        "tool_calls": "\033[94m",  # Blue
        "function_call": "\033[94m",
        "tool_use": "\033[94m",
        "null": "\033[90m",  # Grey
    }
    reset_color = "\033[0m"
    if reason is None:
        return f"\033[90mFinish reason: unknown{reset_color}"
    color = colors.get(reason, "\033[90m")  # Default to grey
    return f"{color}Finish reason: {reason}{reset_color}"

def query_server(
    prompt: str | list[dict],
    system_prompt: str = "You are a helpful assistant",
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 50,
    max_tokens: int = 128,
    num_completions: int = 1,
    server_port: int = 30000,
    server_address: str = "localhost",
    server_type: str = "sglang",
    model_name: str = "default",
    is_reasoning_model: bool = True,
    budget_tokens: int = 0,
    reasoning_effort: str = "medium",
    log_path: Optional[str] = None,
    call_type: str = "unknown",
    round_idx: int = -1,
):
    match server_type:
        case "local":
            llm = get_llm(model_name)  # legacy fallback
            model = model_name

        case "vllm":
            llm = get_llm(model_name, server_url=f"http://{server_address}:{server_port}/v1")
            model = model_name

        case "sglang":
            from openai import OpenAI
            url = f"http://{server_address}:{server_port}"
            client = OpenAI(api_key=SGLANG_KEY, base_url=f"{url}/v1", timeout=None, max_retries=0)
            model = "default"

        case "deepseek":
            from openai import OpenAI
            client = OpenAI(
                api_key=DEEPSEEK_KEY,
                base_url="https://api.deepseek.com",
                timeout=10000000,
                max_retries=3,
            )
            model = model_name

        case "fireworks":
            from openai import OpenAI
            client = OpenAI(
                api_key=FIREWORKS_API_KEY,
                base_url="https://api.fireworks.ai/inference/v1",
                timeout=10000000,
                max_retries=3,
            )
            model = model_name

        case "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            model = model_name

        case "google":
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_KEY)
            model = model_name

        case "together":
            from together import Together
            client = Together(api_key=TOGETHER_KEY)
            model = model_name

        case "sambanova":
            from openai import OpenAI
            client = OpenAI(api_key=SAMBANOVA_API_KEY, base_url="https://api.sambanova.ai/v1")
            model = model_name

        case "openai":
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_KEY)
            model = model_name

        case _:
            raise NotImplementedError(f"Unsupported server_type: {server_type}")

    # ------------------ Local / vLLM --------------------
    if server_type in {"local", "vllm"}:
        assert isinstance(prompt, str), "Only string prompt supported for local/vllm model"
        cfg = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        output = llm.chat(
            system_prompt,         
            prompt,
            cfg,
        )
        return output

    # ------------------ Cloud APIs ---------------------
    outputs = []

    if server_type == "google":
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_tokens,
            "response_mime_type": "text/plain",
        }
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
            generation_config=generation_config,
        )
        response = model.generate_content(prompt)

        # Usage logging
        usage_metadata = getattr(response, 'usage_metadata', None)
        if usage_metadata:
            input_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
            output_tokens = getattr(usage_metadata, 'candidates_token_count', 0)
            total_tokens = getattr(usage_metadata, 'total_token_count', 0)
            usage_str = f"Usage: In={input_tokens}, Out={output_tokens}, Total={total_tokens}"
            print(usage_str)
            if log_path and log_path != "":
                try:
                    import os
                    import datetime
                    file_exists = os.path.exists(log_path)
                    with open(log_path, "a", encoding="utf-8") as f:
                        if not file_exists:
                            f.write("timestamp,round_idx,call_type,input_tokens,output_tokens,total_tokens\n")
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"{timestamp},{round_idx},{call_type},{input_tokens},{output_tokens},{total_tokens}\n")
                except Exception as e:
                    print(f"Warning: Failed to write usage log to {log_path}: {e}")

        # Finish reason
        try:
            candidates = getattr(response, 'candidates', [])
            if candidates:
                candidate = candidates[0]
                finish_reason_obj = getattr(candidate, 'finish_reason', None)
                finish_reason = getattr(finish_reason_obj, 'name', str(finish_reason_obj))
                print(colorize_finish_reason(finish_reason))
                if finish_reason in {"MAX_TOKENS", "length", "max_tokens"}:
                    print(f"Warning: Output truncated due to max_tokens limit ({max_tokens})")
        except Exception:
            pass

        return response.text

    elif server_type == "anthropic":
        assert isinstance(prompt, str)
        if is_reasoning_model:
            response = client.beta.messages.create(
                model=model,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                thinking={"type": "enabled", "budget_tokens": budget_tokens},
                betas=["output-128k-2025-02-19"],
            )
        else:
            response = client.messages.create(
                model=model,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
            )
        # Usage Logging
        if hasattr(response, 'usage'):
            input_tokens = getattr(response.usage, "input_tokens", None)
            output_tokens = getattr(response.usage, "output_tokens", None)
            total_tokens = getattr(response.usage, 'total_tokens', input_tokens + output_tokens
                                   if input_tokens is not None and output_tokens is not None
                                   else input_tokens if output_tokens is None
                                   else output_tokens if input_tokens is None
                                   else None)
            usage_str = f"Usage: In={input_tokens}, Out={output_tokens}, Total={total_tokens}"
            print(usage_str)
            if log_path and log_path != "":
                try:
                    import os
                    import datetime
                    file_exists = os.path.exists(log_path)
                    with open(log_path, "a", encoding="utf-8") as f:
                        if not file_exists:
                            f.write("timestamp,round_idx,call_type,input_tokens,output_tokens,total_tokens\n")
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"{timestamp},{round_idx},{call_type},{input_tokens},{output_tokens},{total_tokens}\n")
                except Exception as e:
                    print(f"Warning: Failed to write usage log to {log_path}: {e}")

        outputs = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text is not None:
                outputs.append(text)
                continue

            block_type = getattr(block, "type", "unknown")
            block_name = getattr(block, "name", "")
            extra = f" ({block_name})" if block_name else ""
            print(f"Skipping non-text {server_type} content block of type '{block_type}'{extra}")

        finish_reason = getattr(response, "stop_reason", None)
        print(colorize_finish_reason(finish_reason))
        if finish_reason in {"length", "max_tokens"}:
            print(f"Warning: Output truncated due to max_tokens limit ({max_tokens})")

    else:
        if isinstance(prompt, str):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = prompt

        if is_reasoning_model and server_type == "openai":
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                reasoning_effort=reasoning_effort,
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p,
            )
        outputs = []
        for choice in response.choices:
            print(colorize_finish_reason(choice.finish_reason))

            if choice.finish_reason == "length":
                print(f"Warning: Output truncated due to max_tokens limit ({max_tokens})")
            outputs.append(choice.message.content)

        if hasattr(response, 'usage') and response.usage:
            input_tokens = getattr(response.usage, "prompt_tokens", getattr(response.usage, "input_tokens", 0))
            output_tokens = getattr(response.usage, "completion_tokens", getattr(response.usage, "output_tokens", 0))
            total_tokens = getattr(response.usage, 'total_tokens', input_tokens + output_tokens)

            usage_str = f"Usage: In={input_tokens}, Out={output_tokens}, Total={total_tokens}"
            print(usage_str)
            if log_path and log_path != "":
                try:
                    import os
                    import datetime
                    file_exists = os.path.exists(log_path)
                    with open(log_path, "a", encoding="utf-8") as f:
                        if not file_exists:
                            f.write("timestamp,round_idx,call_type,input_tokens,output_tokens,total_tokens\n")
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"{timestamp},{round_idx},{call_type},{input_tokens},{output_tokens},{total_tokens}\n")
                except Exception as e:
                    print(f"Warning: Failed to write usage log to {log_path}: {e}")

    return outputs[0] if len(outputs) == 1 else outputs
