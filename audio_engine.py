import threading
import time
import pyttsx3
import speech_recognition as sr

trigger_training_name = None
is_speaking = False


def speak(text, wait=False):
    """Ez a függvény indítja el a beszédet egy külön háttérszálon (nem blokkoló)
    text: a mondandó
    wait: ha True, megvárja a végét (fagy a kép), ha False, háttérben fut (folyamatos kép)
    """
    global is_speaking

    def run_speech():
        """Elindítja a beszédet egy háttérszálon vagy blokkolva."""
        global is_speaking
        try:
            is_speaking = True
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')

            for voice in voices:
                if "English" in voice.name or "en-US" in voice.id:
                    engine.setProperty('voice', voice.id)
                    break
            engine.setProperty('rate', 180)  # beszedsebesség
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
        except Exception as e:
            print(f"Hiba a beszednel: {e}")
        finally:
            time.sleep(0.2)
            is_speaking = False
    if wait:
        run_speech()
    else:
        threading.Thread(target=run_speech).start()  # Háttérben fut


def continuous_audio_listener():
    """Folyamatosan figyel a háttérben az ébresztési szóra."""

    global trigger_training_name, is_speaking
    recogniser = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recogniser.adjust_for_ambient_noise(source=source, duration=0.8)

    print("-> Siri ébresztési szó figyelése elindult a háttérben...")
    while True:
        # Ha a főszál éppen ment egy arcot, várunk, nem szakítjuk félbe
        if trigger_training_name is not None or is_speaking:
            time.sleep(0.2)
            continue
        with mic as source:
            try:
                # 4s ig figyel
                audio = recogniser.listen(
                    source=source, timeout=2, phrase_time_limit=4)
                text = recogniser.recognize_google(
                    audio, language="en-US").lower()

                if "siri" in text:
                    # print("[HÁTTÉRSZÁL] Siri aktiválva!")
                    if "my name is" in text:
                        extracted_name = text.split(
                            "my name is")[-1].strip()  # kiszedjuk a nevet
                        extracted_name = (extracted_name.replace(
                            ".", "").replace("?", "").strip())
                        if extracted_name:
                            # print(f"[HÁTTÉRSZÁL] Név bekészítve mentésre: {extracted_name}")
                            trigger_training_name = (extracted_name)

            except (sr.WaitTimeoutError, sr.UnknownValueError):
                # Ha csend van vagy nem érthető a zaj, hibajelzés nélkül megy tovább a háttér ciklus
                continue
            except Exception as e:
                print(f"Beszéd hiba a háttérben: {e}")
                time.sleep(1)


def start_audio_thread():
    """Segédfüggvény a háttérszál egyszerű indításához a main-ből."""
    audio_thread = threading.Thread(
        target=continuous_audio_listener, daemon=True)
    audio_thread.start()
