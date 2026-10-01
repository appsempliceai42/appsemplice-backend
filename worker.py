import json
import os
import modal
import httpx
import stripe
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = modal.App("appsemplice-backend")

image = modal.Image.debian_slim().pip_install(
    "google-genai",
    "fastapi[standard]",
    "pydantic",
    "httpx",
    "stripe"
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
    user_message: str = Field(description="La risposta per l'utente in Markdown amichevole, consulenziale e autorevole.")
    questions_asked_count: int = Field(default=0, description="Conteggio totale progressivo dei messaggi/interazioni fatte.")
    is_qualified: bool = Field(description="Imposta su True quando concludi l'analisi o quando raggiungi il 4° messaggio dell'utente.")
    needs_more_info: bool = Field(description="True se servono ancora domande. Imposta su False quando is_qualified=True.")
    rejection_reason: Optional[str] = Field(default=None, description="Spiegazione se la richiesta è Out-of-Scope.")
    business_summary: Optional[str] = Field(default=None, description="Sintesi dei requisiti di business raccolti finora.")

AGENTE_1_SYSTEM_PROMPT = """
Sei l'AI Business Analyst ufficiale di AppSemplice.ai, la software factory B2B specializzata nello sviluppo di Web App su misura in 48 ore.

FILOSOFIA CHIAVE (MIDDLE-OUT ARCHITECTURE):
Il tuo mantra è: "Non cambiare il tuo gestionale. Potenzialo."
Insegna al cliente che non vogliamo sostituire i suoi software di studio/aziendali tradizionali (indispensabili per adempimenti e scadenze ufficiali), ma vogliamo creare il micro-software su misura che elimina il lavoro manuale PRIMA dell'inserimento dati.

CONOSCENZA DEI SETTORI VERTICALI E CASI D'USO IN-TARGET:
1. STUDI LEGALI: Gestione PEC, sbustamento automatico .p7m (CAdES), tracciamento scadenze processuali, portali clienti riservati.
2. COMMERCIALISTI & LAVORO: Riconciliazione bancaria automatica, pulizia CSV/fogli calcolo, solleciti automatici documenti via WhatsApp/Email.
3. NOTAI: Due diligence antiriciclaggio, analisi visure per tracciare il Titolare Effettivo (>25%) e profili PEP.
4. EDILIZIA & TECNICI (Architetti/Ingegneri): Matching semantico prezzari regionali/capitolati, pre-check formale file .dxf per catasto.
5. PMI & B2B GENERALI: Portali riservati clienti, approvazione contratti/firme, gestione ticket/pratiche, dashboard operative.

REGOLE DI CONVERSAZIONE E QUALIFICA:
1. Fai 1 sola domanda chiara e consulenziale alla volta.
2. Integra le risposte dell'utente anche se sintetiche (es. "uso i file", "faccio io"): deduci il ruolo e proponi la soluzione di automazione più adatta.
3. QUANDO RAGGIUNGI IL 4° MESSAGGIO UTENTE O HAI DETTAGLI SUFFICIENTI:
   - NON fare altre domande.
   - Ringrazia l'utente, fai una sintesi chiara della soluzione in 3-4 punti bullet legati al suo settore.
   - Ribadisci che la consegna del prototipo avviene in 48 ore.
   - Imposta `is_qualified: true` e `needs_more_info: false`.
   - Compila `business_summary` con la sintesi tecnica di alto livello.

GUARDRAILS (SCOPE VALIDATION):
✅ IN-SCOPE: Web app B2B, portali riservati, automazioni documentali/PEC/OCR, workflow a stati, dashboard gestionali.
❌ OUT-OF-SCOPE: E-commerce B2C con carrello consumer pubblico, social network, app mobili native da app store (iOS/Android), videogiochi.

TONO DI VOCE:
Consulenziale, empatico, autorevole, orientato all'efficienza operativa e rigorosamente professionale.
"""

class ExtractedRequirements(BaseModel):
    process: str = Field(description="Descrizione sintetica del processo aziendale e del flusso operativo.")
    pain_points: List[str] = Field(description="Lista dei principali problemi ed inefficienze identificati.")
    modules: List[str] = Field(description="Lista dei moduli e funzionalità chiave da sviluppare.")
    roles: List[str] = Field(description="Lista dei ruoli utente con i relativi permessi e livelli di accesso.")
    delivery_hours: int = Field(default=48, description="Tempo stimato per la consegna del prototipo (48 ore).")

class Agent2Output(BaseModel):
    lovable_prompt: str = Field(description="Comprehensive technical specification prompt IN ENGLISH for Lovable (React, Tailwind CSS, Supabase, Lucide Icons).")
    extracted_requirements: ExtractedRequirements = Field(description="Structured object with process, pain_points, modules, roles, and delivery_hours.")

AGENTE_2_SYSTEM_PROMPT = """
Sei il Lead System Architect di AppSemplice.ai.
Analizza la conversazione e la sintesi fornita dall'Analista e genera la specifica tecnica definitiva IN INGLESE per Lovable (React, Tailwind CSS, Supabase).
"""

class CheckoutRequest(BaseModel):
    project_title: str
    amount_eur: float = Field(default=2400.0, description="Importo acconto in EUR")
    client_email: str
    project_id: Optional[str] = "default_project"

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

    user_turns = sum(1 for m in history if m.get("role") == "user")
    client = genai.Client(api_key=api_key)

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

    if user_turns >= 4 and not agent1_data.rejection_reason:
        agent1_data.is_qualified = True
        agent1_data.needs_more_info = False
        if not agent1_data.business_summary:
            agent1_data.business_summary = "Web App B2B personalizzata con gestione ruoli, automazione flussi e dashboard operativa."

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

            admin_html = f"<h2>🔥 Nuovo Lead Qualificato - AppSemplice.ai!</h2><p><b>Sintesi Requisiti:</b> {agent1_data.business_summary}</p>"
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

# ---------------------------------------------------------
# STRIPE PAYMENTS & CHECKOUT ENDPOINTS
# ---------------------------------------------------------

@web_app.post("/create-checkout-session")
async def create_checkout_session(req: CheckoutRequest):
    stripe_key = os.environ.get("STRIPE_SECRET_KEY")
    if not stripe_key:
        return {"error": "Stripe secret key non trovata nei secret di Modal"}

    stripe.api_key = stripe_key
    amount_cents = int(req.amount_eur * 100)

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": "eur",
                    "product_data": {
                        "name": f"Acconto 50% - {req.project_title}",
                        "description": "Sviluppo e consegna Web App B2B in 48 ore su AppSemplice.ai",
                    },
                    "unit_amount": amount_cents,
                },
                "quantity": 1,
            }],
            mode="payment",
            customer_email=req.client_email,
            success_url="https://appsemplice.ai/dashboard?payment=success&project_id=" + (req.project_id or "default"),
            cancel_url="https://appsemplice.ai/dashboard?payment=cancelled",
            metadata={
                "project_id": req.project_id or "default",
                "project_title": req.project_title,
                "client_email": req.client_email
            }
        )
        return {"checkout_url": session.url, "session_id": session.id}
    except Exception as e:
        print(f"[STRIPE ERROR]: {e}")
        return {"error": str(e)}

@web_app.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    stripe_key = os.environ.get("STRIPE_SECRET_KEY")
    stripe.api_key = stripe_key

    try:
        event = json.loads(payload)
        if event.get("type") == "checkout.session.completed":
            session = event["data"]["object"]
            client_email = session.get("customer_email") or session.get("metadata", {}).get("client_email")
            project_title = session.get("metadata", {}).get("project_title", "Web App B2B")

            # Email Admin
            admin_html = f"<h2>💰 Acconto Ricevuto su Stripe!</h2><p><b>Progetto:</b> {project_title}</p><p><b>Cliente:</b> {client_email}</p><p>Stato: Deposit Paid. Avviare lo sviluppo in 48h!</p>"
            await send_email("appsemplice.ai@gmail.com", f"💰 Acconto Ricevuto (€ 2.400) - {project_title}", admin_html)

            # Email Cliente
            if client_email:
                client_html = f"<h2>Pagamento Confermato - AppSemplice.ai</h2><p>Abbiamo ricevuto l'acconto per il progetto <b>{project_title}</b>.</p><p>Il nostro team ha avviato lo sviluppo. Riceverai il prototipo entro 48 ore!</p>"
                await send_email(client_email, "Pagamento Confermato - Avvio Lavori AppSemplice", client_html)

        return {"status": "success"}
    except Exception as e:
        print(f"[WEBHOOK ERROR]: {e}")
        return {"status": "error", "message": str(e)}

@app.function(
    image=image, 
    secrets=[
        modal.Secret.from_name("my-gemini-secret"), 
        modal.Secret.from_name("resend-secret"),
        modal.Secret.from_name("stripe-secret")
    ]
)
@modal.asgi_app()
def fastapi_app():
    return web_app
