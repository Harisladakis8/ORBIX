from flask import Blueprint, jsonify, request
from openai import OpenAI, OpenAIError

from config_openai import API_KEY

openai_bp = Blueprint("openai_bp", __name__, url_prefix="/api")
client = OpenAI(api_key=API_KEY)

DEFAULT_INSTRUCTIONS = """You are ORBIX, a helpful assistant.
Use emojis naturally to make responses engaging and friendly.
"""

MAX_MESSAGE_FILES = 10
MAX_CONTEXT_FILES = 8
MAX_TOTAL_FILE_BYTES = 50 * 1024 * 1024


def validate_files(value, label, limit):
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    if len(value) > limit:
        raise ValueError(f"You can send up to {limit} {label}.")

    files = []
    for file in value:
        if not isinstance(file, dict):
            raise ValueError(f"Invalid item in {label}.")

        name = str(file.get("name", "")).strip()[:255]
        mime_type = str(file.get("type", "application/octet-stream")).lower()
        file_data = file.get("data")
        size = file.get("size", 0)

        if not name or not isinstance(file_data, str):
            raise ValueError(f"Invalid file in {label}.")
        if not file_data.startswith("data:") or ";base64," not in file_data:
            raise ValueError(f"{name} is not a valid Base64 file.")
        if not isinstance(size, (int, float)) or size < 0:
            raise ValueError(f"Invalid size for {name}.")

        files.append({
            "name": name,
            "type": mime_type,
            "size": int(size),
            "data": file_data,
        })

    return files


def add_file_group(content, label, files):
    if not files:
        return

    names = ", ".join(file["name"] for file in files)
    content.append({"type": "input_text", "text": f"{label}: {names}"})

    for file in files:
        if file["type"].startswith("image/"):
            content.append({
                "type": "input_image",
                "image_url": file["data"],
                "detail": "auto",
            })
        else:
            content.append({
                "type": "input_file",
                "filename": file["name"],
                "file_data": file["data"],
            })

@openai_bp.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    instruction = data.get("instruction", "").strip()
    provider = data.get("provider", "gpt").lower()
    history = data.get("history", [])

    try:
        message_files = validate_files(
            data.get("message_files", data.get("files", [])),
            "message files",
            MAX_MESSAGE_FILES,
        )
        context_files = validate_files(
            data.get("context_files", []),
            "Prompt Box files",
            MAX_CONTEXT_FILES,
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    total_file_bytes = sum(
        file["size"] for file in message_files + context_files
    )
    if total_file_bytes >= MAX_TOTAL_FILE_BYTES:
        return jsonify({"error": "The total file size must be less than 50 MB."}), 400

    if not message:
        return jsonify({"error": "Message is required"}), 400

    if provider != "gpt":
        return jsonify({"error": "Gemini backend is not configured yet"}), 400

    instructions = DEFAULT_INSTRUCTIONS
    if instruction:
        instructions = f"""{DEFAULT_INSTRUCTIONS}

Custom application instruction:
<custom_instruction>
{instruction}
</custom_instruction>

Follow the custom application instruction exactly. If it conflicts with the
default identity, tone, or style above, the custom instruction takes priority.
"""

    input_messages = []
    if isinstance(history, list):
        for item in history[-20:]:
            if not isinstance(item, dict):
                continue

            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                input_messages.append({"role": role, "content": content.strip()})

    user_content = [{"type": "input_text", "text": message}]

    # Prompt Box files are persistent context; message files belong to this turn.
    add_file_group(user_content, "Persistent Prompt Box files", context_files)
    add_file_group(user_content, "Files attached to the current message", message_files)

    input_messages.append({"role": "user", "content": user_content})

    try:
        response = client.responses.create(
            model="gpt-4.1",
            instructions=instructions,
            input=input_messages,
        )
    except OpenAIError as error:
        return jsonify({"error": str(error)}), 502

    return jsonify({"text": response.output_text})
