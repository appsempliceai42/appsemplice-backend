import json
import os
import modal
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = modal.App("appsemplice-backend")

image = modal.Image.debian_slim().pip_install(
    "google-genai",
    "fastapi[standard]"
)

# Memoria di sessione volatile per mantenere lo storico della chat
SESSIONS = {}

SYSTEM_PROMPT = """
Sei l'AI Business Analyst ufficiale di "AppSemplice.ai". Il tuo compito è qualificare i lead B2B (avvocati, commercialisti, consulenti, PMI) e raccogliere i requisiti per creare la loro Web App personalizzata in 48 ore.

REGOLE DI CONVERSAZIONE:
1. TONO: Empatico, pragmatico, estremamente professionale e chiaro. Niente gergo tecnico. Fai massimo 1 o 2 domande mirate alla volta per capire flussi, utenti e dati in ingresso/uscita.
2. GUARDRAILS (Filtri di qualificazione):
   - Accetta SOLO richieste per: portali clienti, gestionali, onboarding, upload documenti, dashboard.
   - RIFIUTA progetti fuori target (e-commerce, app native, giochi).
   - Se fuori target, imposta "is_qualified": false, "needs_more_info": false e valorizza "rejection_reason" con: "In AppSemplice ci focalizziamo esclusivamente su Web App gestionali e portali operativi per professionisti per garantire la consegna in 48 ore. La tua richiesta richiede un'infrastruttura diversa da quella che gestiamo."
3. TRIGGER DI CHIUSURA (Lead Qualificato):
   - Non appena hai compreso Input e Output (in 3-5 scambi), imposta "is_qualified": true e "needs_more_info": false.
   - Genera in "lovable_prompt" un prompt dettagliato in inglese per la creazione del prototipo su Lovable.

FORMATO DI RISPOSTA OBLIGATORIO (JSON VALIDO):
{
  "user_message": "Testo per l'utente in Markdown",
  "is_qualified": false,
  "needs_more_info": true,
  "rejection_reason": null,
  "lovable_prompt": null,
  "extracted_requirements": {
    "process": "Descrizione del processo",
    "pain_points": ["Punto 1", "Punto 2"]
  }
}
"""

web_app = FastAPI()

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@web_app.post("/chat")
async def chat_endpoint(request: Request):
    from google import genai
    from google.genai import types
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {
            "message": "Errore configurazione API Key.",
            "is_qualified": False,
            "needs_more_info": True,
            "rejection_reason": None,
            "lovable_prompt": None,
            "extracted_requirements": None
        }

    try:
        data = await request.json()
    except Exception:
        data = {}

    session_id = data.get("session_id", "default_session")
    user_message = data.get("message", "").strip()

    if not user_message:
        return {
            "message": "Errore: nessun messaggio inviato.",
            "is_qualified": False,
            "needs_more_info": True,
            "rejection_reason": None,
            "lovable_prompt": None,
            "extracted_requirements": None
        }

    # Recupera o crea lo storico di sessione
    if session_id not in SESSIONS:
        SESSIONS[session_id] = []

    history = SESSIONS[session_id]
    history.append({"role": "user", "parts": [{"text": user_message}]})

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3
            )
        )
        raw_response = response.text
    except Exception as e:
        print(f"Errore Gemini API: {e}")
        return {
            "message": "Scusami, ho avuto un attimo di esitazione. Puoi ripetere l'ultimo concetto?",
            "is_qualified": False,
            "needs_more_info": True,
            "rejection_reason": None,
            "lovable_prompt": None,
            "extracted_requirements": None
        }

    try:
        clean_json = raw_response.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_json)

        # Salva la risposta dell'AI nello storico
        ai_text = parsed.get("user_message", raw_response)
        history.append({"role": "model", "parts": [{"text": ai_text}]})

        return {
            "message": ai_text,
            "is_qualified": parsed.get("is_qualified", False),
            "needs_more_info": parsed.get("needs_more_info", True),
            "rejection_reason": parsed.get("rejection_reason", None),
            "lovable_prompt": parsed.get("lovable_prompt", None),
            "extracted_requirements": parsed.get("extracted_requirements", None)
        }
    except Exception as e:
        print(f"Errore parsing JSON: {e}")
        history.append({"role": "model", "parts": [{"text": raw_response}]})
        return {
            "message": raw_response,
            "is_qualified": False,
            "needs_more_info": True,
            "rejection_reason": None,
            "lovable_prompt": None,
            "extracted_requirements": None
        }

@app.function(image=image, secrets=[modal.Secret.from_name("my-gemini-secret")])
@modal.asgi_app()
def fastapi_app():
    return web_app
