import ollama
import json


def parse_command(user_name, command_text):
    """Beküldi a magyar mondatot az LLM-nek, ellenőrzi a jelszót, és JSON-t ad vissza."""

    system_prompt = """
    Te egy okosotthon vezérlő rendszer vagy.
    A felhasználó magyar mondatából kinyered az eszközt és a cselekvést.

    ÉRVÉNYES ESZKÖZÖK (csak ezek közül választhatsz, pontosan így írva):
    - "kapu" (ha kapuról, nagykapuról, bejáróról van szó)
    - "redőny" (ha redőnyről, árnyékolóról van szó)
    - "lámpa" (ha lámpáról, világításról van szó)
    - "none" (ha nem okosotthon parancs)

    ÉRVÉNYES CSELEKVÉSEK (csak ezek közül választhatsz):
    - "open" (nyitás, felhúzás, beengedés)
    - "close" (zárás, leengedés)
    - "on" (bekapcsolás)
    - "off" (kikapcsolás)
    - "unknown" (ha nem egyértelmű)

    SZABÁLYOK:
    1. Csak valid JSON-t adj vissza, semmi mást!
    2. Pontosan ezeket a kulcsokat használd: "user", "device", "action", "auth"
    3. A "device" CSAK a fenti listából kerülhet ki, soha ne találj ki új eszköznevet!
    4. Az "auth" mindig true legyen ismert felhasználónál.

    PÉLDÁK:
    "zeusz nyisd ki a kaput" -> {"user": "Robi", "device": "kapu", "action": "open", "auth": true}
    "zeusz engedd be a kocsit" -> {"user": "Robi", "device": "kapu", "action": "open", "auth": true}
    "zeusz húzd le a redőnyt" -> {"user": "Robi", "device": "redőny", "action": "close", "auth": true}
    """

    try:
        response = ollama.chat(model='qwen2.5:1.5b', messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Felhasználó: {user_name}. Mondat: {command_text}"}
        ], format='json')

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
