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

SESSIONS: Dict[str, List[Dict[str, Any]]] = {}

# ---------------------------------------------------------
# FUNZIONE DI INVIO EMAIL (RESEND API)
# ---------------------------------------------------------
async def send_email(to_email: str, subject: str, html_content: str):
    resend_api_key = os.environ.get("RESEND_API_KEY")
    if not resend_api_key:
        print("[EMAIL WARNING] RESEND_API_KEY non configurata nei secret Modal.")
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
            print(f"[EMAIL STATUS] {response.status_code}: {response.text}")
            return response.status_code in [200, 201]
        except Exception as e:
            print(f"[EMAIL ERROR] Fallito invio a {to_email}: {e}")
            return False

# ---------------------------------------------------------
# SCHEMI E SCHEDE AGENTI
# ---------------------------------------------------------

class Agent1Output(BaseModel):
    user_message: str = Field(description="La risposta per l'utente in Markdown amichevole e consulenziale.")
    questions_asked_count: int = Field(description="Conteggio totale progressivo delle domande chiave fatte finora nella conversazione.")
    is_qualified: bool = Field(description="Imposta su True SOLO SE hai posto ed ottenuto risposta ad ALMENO 4 DOMANDE chiave E hai dettagli sufficienti per lo sviluppo.")
    needs_more_info: bool = Field(description="True se servono ancora domande di chiarimento (obbligatorio se questions_asked_count < 4, con limite massimo a 8).")
    rejection_reason: Optional[str] = Field(default=None, description="Spiegazione se la richiesta è Out-of-Scope (es. e-commerce B2C, app native mobile, social network).")
    business_summary: Optional[str] = Field(default=None, description="Sintesi dei requisiti di business raccolti finora (compilato quando is_qualified=True).")

AGENTE_1_SYSTEM_PROMPT = """
Sei l'AI Business Analyst ufficiale di AppSemplice.ai.
Il tuo obiettivo è dialogare con clienti B2B (professionisti, PMI, consulenti), comprendere i loro problemi operativi e qualificare il lead per la realizzazione di una Web App gestionale/portale operativo in 48 ore.

REGOLE RIGIDE SUL NUMERO DI DOMANDE (MINIMO 4, MASSIMO 8):
1. **REQUISITO MINIMO (Almeno 4 Domande)**:
   - NON PUOI MAI impostare `is_qualified: true` finché non hai fatto ed ottenuto risposta ad **ALMENO 4 DOMANDE chiave distinte**.
   - Fai 1 o massimo 2 domande alla volta per mantenere la conversazione fluida.
2. **REQUISITO MASSIMO (Massimo 8 Domande)**:
   - Entro la 8ª domanda devi concludere l'analisi e decidere se qualificare (`is_qualified: true`) o rifiutare.
3. **AREE CHIAVE DA ESPLORARE (4 Domande Fondamentali)**:
   - Domanda 1: Processo aziendale e flusso operativo (Input ➔ Passaggi ➔ Output attesi).
   - Domanda 2: Tipologie di utenti e permessi (es. Admin vs Staff vs Cliente finale).
   - Domanda 3: Gestione dati e documenti (es. upload PDF/Excel, form di inserimento, export).
   - Domanda 4: Problema principale (pain point) da risolvere e obiettivo primario.

GUARDRAILS (SCOPE VALIDATION):
- **IN-TARGET**: Portali clienti, gestionali interni, onboarding, dashboard operative, upload/gestione documenti, automazione pratiche B2B.
- **OUT-OF-SCOPE**: E-commerce B2C con carrello/catalogo pubblico consumer, app mobili native (iOS/Android da store), social media, videogiochi.
- **Se OUT-OF-SCOPE**: Puoi interrompere la conversazione anche prima delle 4 domande. Imposta `is_qualified: false`, `needs_more_info: false` e spiega in `rejection_reason` cosa realizziamo in 48 ore.

TONO DI VOCE:
- Consulenziale, professionale, empatico e privo di gergo tecnico complesso.
"""

class Agent2Output(BaseModel):
    lovable_prompt: str = Field(description="Comprehensive technical specification prompt in English for Lovable (React, Tailwind, Supabase).")
    extracted_requirements: Dict[str, Any] = Field(description="Structured dictionary with process, pain_points, modules, roles, and delivery_hours.")

AGENTE_2_SYSTEM_PROMPT = """
Sei il Lead System Architect di AppSemplice.ai.
Analizza la conversazione completata dall'AI Business Analyst e genera la specifica tecnica definitiva per Lovable.

REQUISITI DI OUTPUT:
1. lovable_prompt: Un prompt dettagliato e professionale IN INGLESE strutturato per Lovable (React, Tailwind, Supabase).
2. extracted_requirements: Un oggetto JSON con process, pain_points, modules, roles e delivery_hours (48).
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
            "message": "Nessun messaggio inviato.",
            "is_qualified": False,
            "needs_more_info": True,
            "rejection_reason": None,
            "lovable_prompt": None,
            "extracted_requirements": None
        }

    if session_id not in SESSIONS:
        SESSIONS[session_id] = []

    history = SESSIONS[session_id]
    history.append({"role": "user", "parts": [{"text": user_message}]})

    client = genai.Client(api_key=api_key)

    try:
        res1 = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=AGENTE_1_SYSTEM_PROMPT,
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=Agent1Output,
            )
        )
        agent1_data: Agent1Output = Agent1Output.model_validate_json(res1.text)
    except Exception as e:
        print(f"Errore Agente 1: {e}")
        return {
            "message": "Scusami, ho avuto un piccolo problema di connessione. Puoi ripetere?",
            "is_qualified": False,
            "needs_more_info": True,
            "rejection_reason": None,
            "lovable_prompt": None,
            "extracted_requirements": None
        }

    history.append({"role": "model", "parts": [{"text": agent1_data.user_message}]})

    lovable_prompt = None
    extracted_requirements = None

    if agent1_data.is_qualified:
        try:
            architect_input = (
                f"Sintesi Business dall'Analista: {agent1_data.business_summary}\n\n"
                f"Storico Completo Conversazione:\n{json.dumps(history, indent=2)}"
            )
            
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
            extracted_requirements = agent2_data.extracted_requirements

            # Notifica automatica all'Admin per nuovo lead qualificato
            admin_html = f"""
            <h2>🔥 Nuovo Lead Qualificato su AppSemplice.ai!</h2>
            <p><b>Sintesi Requisiti:</b> {agent1_data.business_summary}</p>
            <p>Accedi all'Admin Panel per revisionare la richiesta e pubblicare il preventivo.</p>
            """
            await send_email("appsemplice.ai@gmail.com", "🔥 Nuovo Lead Qualificato - AppSemplice", admin_html)

        except Exception as e:
            print(f"Errore Agente 2: {e}")

    return {
        "message": agent1_data.user_message,
        "is_qualified": agent1_data.is_qualified,
        "needs_more_info": agent1_data.needs_more_info,
        "rejection_reason": agent1_data.rejection_reason,
        "lovable_prompt": lovable_prompt,
        "extracted_requirements": extracted_requirements
    }

# Endpoint 1: Invio notifica al cliente per proposta pronta
@web_app.post("/notify-proposal")
async def notify_proposal(request: Request):
    data = await request.json()
    client_email = data.get("client_email")
    project_title = data.get("project_title", "Il tuo Progetto Web App")
    total_price = data.get("total_price", "4.800")
    deposit_amount = data.get("deposit_amount", "2.400")

    if not client_email:
        return {"success": False, "error": "Email cliente mancante"}

    client_html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #e0e0e0; border-radius: 10px;">
        <h2 style="color: #000;">La tua Proposta per {project_title} è Pronta! 🚀</h2>
        <p>Ciao,</p>
        <p>Abbiamo completato l'analisi tecnica e strutturato la proposta per la tua nuova Web App con consegna in <strong>48 ore</strong>.</p>
        
        <div style="background-color: #f9f9f9; padding: 15px; border-radius: 8px; margin: 20px 0;">
            <p style="margin: 5px 0;"><strong>Valore Totale Progetto:</strong> € {total_price}</p>
            <p style="margin: 5px 0;"><strong>Acconto Avvio Lavori (50%):</strong> € {deposit_amount}</p>
            <p style="margin: 5px 0;"><strong>Tempi di Consegna Prototipo:</strong> 48 Ore lavorative</p>
        </div>

        <p>Puoi accedere subito alla tua dashboard personale per visualizzare le specifiche, i mockup e confermare l'ordine:</p>
        
        <a href="https://appsemplice.ai/dashboard" style="display: inline-block; background-color: #000; color: #fff; padding: 12px 25px; text-decoration: none; border-radius: 6px; font-weight: bold; margin-top: 10px;">Visualizza Preventivo e Mockup</a>
        
        <br><br>
        <hr style="border: none; border-top: 1px solid #eee;">
        <p style="font-size: 12px; color: #888;">AppSemplice.ai - Sviluppo Web App B2B in 48h</p>
    </div>
    """

    success = await send_email(
        client_email,
        f"La tua Proposta per {project_title} è pronta! - AppSemplice.ai",
        client_html
    )

    return {"success": success}

@app.function(
    image=image, 
    secrets=[
        modal.Secret.from_name("my-gemini-secret"),
        modal.Secret.from_name("resend-secret")
    ]
)
@modal.asgi_app()
def fastapi_app():
    return web_app
