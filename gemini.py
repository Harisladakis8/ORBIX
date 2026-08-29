import base64
import binascii

from flask import Blueprint, jsonify, request
from google import genai
from google.genai import errors, types

from config_gemini import API_KEY


gemini_bp = Blueprint("gemini_bp", __name__, url_prefix="/api/gemini")
client = genai.Client(api_key=API_KEY)

GEMINI_MODEL = "gemini-2.5-flash"
MAX_MESSAGE_FILES = 10
MAX_CONTEXT_FILES = 8
MAX_TOTAL_FILE_BYTES = 50 * 1024 * 1024

DEFAULT_INSTRUCTIONS = """You are ORBIX, a helpful assistant.
Use emojis naturally to make responses engaging and friendly.
"""


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
        data_url = file.get("data")

        if not name or not isinstance(data_url, str):
            raise ValueError(f"Invalid file in {label}.")
        if not data_url.startswith("data:") or ";base64," not in data_url:
            raise ValueError(f"{name} is not a valid Base64 file.")

        header, encoded_data = data_url.split(",", 1)
        header_mime_type = header[5:].split(";", 1)[0].lower()
        if mime_type == "application/octet-stream" and header_mime_type:
            mime_type = header_mime_type

        try:
            file_bytes = base64.b64decode(encoded_data, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError(f"Could not decode {name}.") from error

        files.append({
            "name": name,
            "type": mime_type,
            "bytes": file_bytes,
        })

    return files


def add_file_group(parts, label, files):
    if not files:
        return

    names = ", ".join(file["name"] for file in files)
    parts.append(types.Part.from_text(text=f"{label}: {names}"))

    for file in files:
        parts.append(types.Part.from_bytes(
            data=file["bytes"],
            mime_type=file["type"],
        ))


@gemini_bp.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    instruction = str(data.get("instruction", "")).strip()
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "Message is required"}), 400

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
        len(file["bytes"]) for file in message_files + context_files
    )
    if total_file_bytes >= MAX_TOTAL_FILE_BYTES:
        return jsonify({"error": "The total file size must be less than 50 MB."}), 400

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

    contents = []
    if isinstance(history, list):
        for item in history[-20:]:
            if not isinstance(item, dict):
                continue

            role = item.get("role")
            content = item.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue
            if not content.strip():
                continue

            contents.append(types.Content(
                role="model" if role == "assistant" else "user",
                parts=[types.Part.from_text(text=content.strip())],
            ))

    user_parts = [types.Part.from_text(text=message)]

    # Prompt Box files are persistent context; message files belong to this turn.
    add_file_group(user_parts, "Persistent Prompt Box files", context_files)
    add_file_group(user_parts, "Files attached to the current message", message_files)

    contents.append(types.Content(role="user", parts=user_parts))

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=instructions,
            ),
        )
    except errors.APIError as error:
        return jsonify({"error": f"Gemini API error: {error}"}), 502

    if not response.text:
        return jsonify({"error": "Gemini returned no text response."}), 502

    return jsonify({"text": response.text})
