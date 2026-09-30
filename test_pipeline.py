import httpx
import json
import time

BASE_URL = "https://appsemplice-ai--appsemplice-backend-fastapi-app.modal.run"
SESSION_ID = f"test_e2e_{int(time.time())}"

def print_separator(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

async def run_e2e_simulation():
    async with httpx.AsyncClient(timeout=30.0) as client:
        print_separator(f"AVVIO SIMULAZIONE E2E - Sessione: {SESSION_ID}")

        # GIRO 1
        print("\n[GIRO 1] Invio idea di business iniziale...")
        res1 = await client.post(f"{BASE_URL}/chat", json={
            "session_id": SESSION_ID,
            "message": "Vorrei un portale B2B per uno studio legale per condividere atti e pratiche con i clienti."
        })
        d1 = res1.json()
        print(f"🤖 AI Response:\n{d1.get('message')}")
        print(f"\n📊 Status -> Qualified: {d1.get('is_qualified')}, Domande poste: {d1.get('questions_asked_count')}")

        time.sleep(1)

        # GIRO 2
        print_separator("GIRO 2: Risposta su Ruoli e Utenti")
        res2 = await client.post(f"{BASE_URL}/chat", json={
            "session_id": SESSION_ID,
            "message": "Accederanno 5 legali dello studio con visibilità totale e i singoli clienti con accesso protetto solo ai loro file."
        })
        d2 = res2.json()
        print(f"🤖 AI Response:\n{d2.get('message')}")
        print(f"\n📊 Status -> Qualified: {d2.get('is_qualified')}, Domande poste: {d2.get('questions_asked_count')}")

        time.sleep(1)

        # GIRO 3
        print_separator("GIRO 3: Risposta su Pain Point e Email")
        res3 = await client.post(f"{BASE_URL}/chat", json={
            "session_id": SESSION_ID,
            "message": "Oggi usiamo le email e perdiamo costantemente le ultime versioni dei file."
        })
        d3 = res3.json()
        print(f"🤖 AI Response:\n{d3.get('message')}")
        print(f"\n📊 Status -> Qualified: {d3.get('is_qualified')}, Domande poste: {d3.get('questions_asked_count')}")

        time.sleep(1)

        # GIRO 4 (Trigger Qualifica - 4 Domande Completate)
        print_separator("GIRO 4: Risposta su Dati/Permessi (Soglia 4 Domande Raggiunta)")
        res4 = await client.post(f"{BASE_URL}/chat", json={
            "session_id": SESSION_ID,
            "message": "I clienti devono poter caricare PDF e scansioni, la segreteria e i legali approvano o richiedono integrazioni."
        })
        d4 = res4.json()
        print(f"🤖 AI Response:\n{d4.get('message')}")
        print(f"\n📊 Status -> Qualified: {d4.get('is_qualified')}, Domande poste: {d4.get('questions_asked_count')}")

        if d4.get("is_qualified"):
            print("\n✅ LEAD QUALIFICATO CON SUCCESSO AL 4° GIRO!")
            lovable_prompt = d4.get("lovable_prompt") or ""
            print(f"\n📄 Lovable Prompt generato ({len(lovable_prompt)} caratteri):")
            print(lovable_prompt[:350] + ("..." if len(lovable_prompt) > 350 else ""))
            print("\n📦 Requisiti Estratti (JSON):")
            print(json.dumps(d4.get("extracted_requirements"), indent=2))
        else:
            print("\n❌ ERRORE: Il lead doveva essere qualificato al 4° giro!")

        time.sleep(1)

        # TEST NOTIFY
        print_separator("TEST ENDPOINT /notify-proposal")
        res_notify = await client.post(f"{BASE_URL}/notify-proposal", json={
            "client_email": "appsemplice.ai@gmail.com",
            "project_title": "Portale Gestione Pratiche Studio Legale",
            "total_price": "4.800",
            "deposit_amount": "2.400"
        })
        print(f"📬 Risposta /notify-proposal: {res_notify.json()}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_e2e_simulation())
