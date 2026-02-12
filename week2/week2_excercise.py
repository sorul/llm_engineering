import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from typing import List, Dict
import random

MODEL = "gpt-4o-mini"
TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"


def get_tools() -> List[Dict]:
  random_word_function = {
      "name":
          "get_random_english_word",
      "description":
          "Si hay una palabra inventada en español, utiliza esta función para devolver una palabra aleatoria en inglés y así ayudar a traducir la palabra inventada. No uses esta función para nada más que para traducir palabras inventadas.",
      "parameters":
          {
              "type": "object",
              "properties":
                  {
                      "invented_word":
                          {
                              "type":
                                  "string",
                              "description":
                                  "La palabra inventada en español que se desea traducir al inglés",
                          },
                  },
              "required": ["invented_word"],
              "additionalProperties": False
          }
  }
  return [{"type": "function", "function": random_word_function}]


def get_system_message():
  return '''
    Eres un asistente que traduce del español al inglés. No respondas a las frases del usuario, limítate a traducir todos los mensajes que te lleguen.
  '''


def chat(history):
  history = history or []
  messages = [{"role": "system", "content": get_system_message()}] + history
  tools = get_tools()
  openai = OpenAI()
  response = openai.chat.completions.create(
      model=MODEL, messages=messages, tools=tools
  )

  if response.choices[0].finish_reason == "tool_calls":
    message = response.choices[0].message

    # Llamamos a la herramienta y obtenemos su resultado solo para la segunda llamada a OpenAI
    tool_response = handle_tool_call(message)

    # Agregamos el contexto previo y la respuesta de la herramienta a los mensajes para la siguiente llamada a OpenAI.
    messages.append(message.model_dump(exclude_none=True))
    messages.append(tool_response)

    # Llamamos al modelo de nuevo con la respuesta de la herramienta para obtener la respuesta final.
    response = openai.chat.completions.create(model=MODEL, messages=messages)

  # La respuesta final es la respuesta del modelo después de llamar a la herramienta (si se llamó).
  reply = response.choices[0].message.content
  history += [{"role": "assistant", "content": reply}]

  print("\n\nHistory:")
  print(json.dumps(history, indent=2, ensure_ascii=False, default=str))
  return history


def get_random_english_word():
  english_words = [
      "serendipity",
      "ephemeral",
      "luminous",
      "quintessential",
      "mellifluous",
  ]
  return random.choice(english_words)


def handle_tool_call(message):
  tool_call = message.tool_calls[0]
  function_name = tool_call.function.name
  arguments = json.loads(tool_call.function.arguments or "{}")

  if function_name == get_random_english_word.__name__:
    content = {
        "invented_word": arguments.get("invented_word"),
        "english_word": get_random_english_word(),
    }
  else:
    content = {"error": f"Tool no soportada: {function_name}"}

  response = {
      "role":
          "tool",
      "content":
          json.dumps(content),
      "tool_call_id":
          tool_call.id
  }
  return response


def transcribe_with_voice_agent(audio_path: str) -> str:
  if not audio_path:
    return ""

  openai = OpenAI()
  with open(audio_path, "rb") as audio_file:
    transcript = openai.audio.transcriptions.create(
        model=TRANSCRIBE_MODEL,
        file=audio_file
    )
  return (transcript.text or "").strip()


if __name__ == "__main__":
  load_dotenv()

  with gr.Blocks() as ui:
    with gr.Row():
      chatbot = gr.Chatbot(height=250)
    with gr.Row():
      entry = gr.Textbox(label="Traduce al ingles.")
    with gr.Row():
      mic = gr.Audio(sources=["microphone"], type="filepath", label="Entrada por voz")
      send_voice = gr.Button("Enviar voz")
    with gr.Row():
      clear = gr.Button("Clear")

    def do_entry(message, history):
      history = history or []
      history += [{"role": "user", "content": message}]
      return "", history

    def do_voice(audio_path, history):
      history = history or []
      text = transcribe_with_voice_agent(audio_path)
      if not text:
        raise gr.Error("No se pudo transcribir el audio.")
      history += [{"role": "user", "content": text}]
      return "", history

    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]) \
        .then(chat, inputs=chatbot, outputs=[chatbot])
    send_voice.click(do_voice, inputs=[mic, chatbot], outputs=[entry, chatbot]) \
        .then(chat, inputs=chatbot, outputs=[chatbot])

    clear.click(lambda: [], inputs=None, outputs=chatbot, queue=False)

# Especifico un puerto porque lo tengo reservado para permitir el micro
ui.launch(server_name="0.0.0.0", server_port=7863, inbrowser=True)
