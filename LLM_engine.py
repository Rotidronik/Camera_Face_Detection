import ollama
import json


def parse_command(user_name, command_text):
    """
    Beküldi a nyers szöveget az LLM-nek, és egy strukturált JSON szótárral tér vissza.
    """
    system_prompt = """
    Te egy okosotthon vezérlő rendszer központi egysége vagy. 
    A feladatod, hogy a felhasználó nyers mondatából kinyerd az irányítani kívánt eszközt és a cselekvést.
    
    SZABÁLYOK:
    1. Csak és kizárólag érvényes JSON formátumban válaszolj!
    2. A JSON struktúrának pontosan ezeket a kulcsokat kell tartalmaznia: 
       "user", "device", "action"
    3. Az "action" csak "open", "close", "on", "off" vagy "unknown" lehet.
    4. Ha nem értelmezhető a parancs, a "device" legyen "none", az "action" pedig "unknown".
    
    PÉLDA VÁLASZ:
    {"user": "Róbert", "device": "kapu", "action": "open"}
    """

    try:
        # Hívás a lokális Ollama szerver felé
        response = ollama.chat(model='llama3', messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Felhasználó: {user_name}. Parancs: {command_text}"}
        ], format='json')

        # A kapott szöveges JSON átalakítása Python Dictionary-vé
        result_dict = json.loads(response['message']['content'])
        return result_dict
    except Exception as e:
        print(f"Hiba az LLM feldolgozás során: {e}")
        return None


# --- SZIMULÁLT TESZTELÉS (Mocking) ---
# Ez a rész csak akkor fut le, ha közvetlenül ezt a fájlt indítod el,
# a main.py-ból történő importáláskor nem!
if __name__ == "__main__":
    print("LLM motor tesztelése...")

    test_commands = [
        "Siri, nyisd ki légyszi a nagykaput, mert hazaértem!",
        "Húzd le a redőnyt a nappaliban.",
        "Milyen idő lesz holnap?"
    ]

    for cmd in test_commands:
        print(f"\nBemenet: '{cmd}'")
        output = parse_command("Róbert", cmd)
        print(f"Kimenet (JSON): {output}")
        if output:
            print(
                f"Feldolgozott adat -> Eszköz: {output.get('device')}, Parancs: {output.get('action')}")
