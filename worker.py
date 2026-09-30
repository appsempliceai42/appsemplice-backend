import json
import os
import modal
import httpx
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = modal.App("appsemplice-backend")

image = modal.Image.debian_slim().pip_install(
    "google-genai",
    "fastapi[standard]",
    "pydantic",
    "httpx"
)

SESSIONS: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------
# FUNZIONE DI INVIO EMAIL (RESEND API)
# ---------------------------------------------------------
async def send_email(to_email: str, subject: str, html_content: str):
    resend_api_key = os.environ.get("RESEND_API_KEY")
    if not resend_api_key:
        return False
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {resend_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "from": "AppSemplice.ai <onboarding@resend.dev>",
                    "to": [to_email],
                    "subject": subject,
                    "html": html_content
                }
            )
            return response.status_code in [200, 201]
        except Exception:
            return False

# ---------------------------------------------------------
# SCHEMI E SCHEDE AGENTI
# ---------------------------------------------------------

class Agent1Output(BaseModel):
    user_message: str = Field(description="La risposta per l'utente in Markdown amichevole e consulenziale.")
    questions_asked_count: int = Field(default=0, description="Conteggio totale progressivo delle domande/interazioni fatte.")
    is_qualified: bool = Field(description="Imposta su True quando concludi l'analisi o quando raggiungi il 4° messaggio dell'utente.")
    needs_more_info: bool = Field(description="True se servono ancora domande. Imposta su False quando is_qualified=True.")
    rejection_reason: Optional[str] = Field(default=None, description="Spiegazione se la richiesta è Out-of-Scope.")
    business_summary: Optional[str] = Field(default=None, description="Sintesi dei requisiti di business raccolti finora.")

AGENTE_1_SYSTEM_PROMPT = """
Sei l'AI Business Analyst ufficiale di AppSemplice.ai.
Il tuo obiettivo è dialogare con clienti B2B, comprendere i loro problemi operativi e qualificare il lead per una Web App B2B in 48 ore.

REGOLE DI CONVERSAZIONE E QUALIFICA:
1. Fai 1 sola domanda chiara alla volta.
2. Integra le risposte anche se l'utente risponde in modo sintetico (es. "io approvo", "lo fa l'assistente"): unisci i pezzi e deduci la soluzione.
3. QUANDO RAGGIUNGI IL 4° MESSAGGIO UTENTE O HAI ABBASTANZA DETTAGLI:
   - NON fare altre domande.
   - Ringrazia l'utente, fai una sintesi chiara della soluzione proposta in 3-4 punti bullet.
   - Imposta `is_qualified: true` e `needs_more_info: false`.
   - Compila `business_summary` con i dettagli raccolti.

GUARDRAILS (SCOPE VALIDATION BINARIO):
✅ IN-SCOPE: Portali B2B, gestione contratti/firme, upload file, gestione ruoli (Admin, Assistente, Cliente), workflow di approvazione, dashboard.
❌ OUT-OF-SCOPE: E-commerce B2C, social network, app native da store, videogiochi.
"""

class ExtractedRequirements(BaseModel):
    process: str = Field(description="Descrizione sintetica del processo aziendale e del flusso operativo.")
    pain_points: List[str] = Field(description="Lista dei principali problemi ed inefficienze identificati.")
    modules: List[str] = Field(description="Lista dei moduli e funzionalità chiave da sviluppare.")
    roles: List[str] = Field(description="Lista dei ruoli utente con i relativi permessi e livelli di accesso.")
    delivery_hours: int = Field(default=48, description="Tempo stimato per la consegna del prototipo (48 ore).")

class Agent2Output(BaseModel):
    lovable_prompt: str = Field(description="Comprehensive technical specification prompt IN ENGLISH for Lovable (React, Tailwind CSS, Supabase).")
    extracted_requirements: ExtractedRequirements = Field(description="Structured object with process, pain_points, modules, roles, and delivery_hours.")

AGENTE_2_SYSTEM_PROMPT = """
Sei il Lead System Architect di AppSemplice.ai.
Analizza la conversazione e genera la specifica tecnica definitiva IN INGLESE per Lovable (React, Tailwind CSS, Supabase).
"""

# ---------------------------------------------------------
# FASTAPI APP & ENDPOINTS
# ---------------------------------------------------------

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
        return {"message": "Errore configurazione API Key.", "is_qualified": False, "needs_more_info": True}

    try:
        data = await request.json()
    except Exception:
        data = {}

    session_id = data.get("session_id", "default_session")
    user_message = data.get("message", "").strip()

    if not user_message:
        return {"message": "Nessun messaggio", "is_qualified": False, "needs_more_info": True}

    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"history": []}

    history = SESSIONS[session_id]["history"]
    history.append({"role": "user", "parts": [{"text": user_message}]})

    # Conteggio reale dei messaggi inviati dall'utente
    user_turns = sum(1 for m in history if m.get("role") == "user")

    client = genai.Client(api_key=api_key)

    # Iniezione dell'Istruzione Forzata se siamo al 4° o successivo messaggio utente
    if user_turns >= 4:
        dynamic_system_instruction = AGENTE_1_SYSTEM_PROMPT + f"\n\n[ISTRUZIONE TASSATIVA DI CHIUSURA]: L'utente ha già inviato {user_turns} messaggi. L'analisi è CONCLUSA. NON FARE ALTRE DOMANDE. Fai la sintesi della soluzione, imposta `is_qualified: true`, `needs_more_info: false` e compila `business_summary`."
    else:
        dynamic_system_instruction = AGENTE_1_SYSTEM_PROMPT + f"\n\n[STATO ATTUALE]: Questo è il messaggio utente numero {user_turns} di 4."

    try:
        res1 = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=dynamic_system_instruction,
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=Agent1Output,
            )
        )
        agent1_data: Agent1Output = Agent1Output.model_validate_json(res1.text)
    except Exception as e:
        print(f"Errore Agente 1: {e}")
        history.pop()
        return {"message": "Errore di connessione temporaneo. Riprova.", "is_qualified": False, "needs_more_info": True}

    # Forzatura lato Python se per qualsiasi motivo l'LLM non avesse impostato la qualifica al 4° turno
    if user_turns >= 4 and not agent1_data.rejection_reason:
        agent1_data.is_qualified = True
        agent1_data.needs_more_info = False
        if not agent1_data.business_summary:
            agent1_data.business_summary = "Portale B2B per gestione e firma contratti con ruoli distinte per Titolare, Assistente e Clienti."

    history.append({"role": "model", "parts": [{"text": agent1_data.user_message}]})

    lovable_prompt = None
    extracted_requirements = None

    if agent1_data.is_qualified:
        try:
            architect_input = f"Sintesi Business: {agent1_data.business_summary}\n\nStorico:\n{json.dumps(history, indent=2)}"
            res2 = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=architect_input,
                config=types.GenerateContentConfig(
                    system_instruction=AGENTE_2_SYSTEM_PROMPT,
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=Agent2Output,
                )
            )
            agent2_data: Agent2Output = Agent2Output.model_validate_json(res2.text)
            lovable_prompt = agent2_data.lovable_prompt
            extracted_requirements = agent2_data.extracted_requirements.model_dump()

            admin_html = f"<h2>🔥 Nuovo Lead Qualificato!</h2><p><b>Sintesi:</b> {agent1_data.business_summary}</p>"
            await send_email("appsemplice.ai@gmail.com", "🔥 Nuovo Lead Qualificato - AppSemplice", admin_html)
        except Exception as e:
            print(f"[ERRORE AGENTE 2]: {e}")

    return {
        "message": agent1_data.user_message,
        "questions_asked_count": user_turns,
        "is_qualified": agent1_data.is_qualified,
        "needs_more_info": agent1_data.needs_more_info,
        "rejection_reason": agent1_data.rejection_reason,
        "lovable_prompt": lovable_prompt,
        "extracted_requirements": extracted_requirements
    }

@web_app.post("/notify-proposal")
async def notify_proposal(request: Request):
    data = await request.json()
    client_email = data.get("client_email")
    if not client_email: return {"success": False}
    html = f"<h2>Preventivo Pronto per {data.get('project_title', 'Web App')}!</h2><p>Totale: € {data.get('total_price', '4.800')}</p><a href='https://appsemplice.ai/dashboard'>Vedi Dashboard</a>"
    success = await send_email(client_email, "Proposta Pronta - AppSemplice", html)
    return {"success": success}

@app.function(
    image=image, 
    secrets=[modal.Secret.from_name("my-gemini-secret"), modal.Secret.from_name("resend-secret")]
)
@modal.asgi_app()
def fastapi_app():
    return web_app
