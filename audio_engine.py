import threading
import time
import speech_recognition as sr
import os
from gtts import gTTS
import pygame

trigger_training_name = None
trigger_llm_command = None
is_speaking = False
pygame.mixer.init()


def speak(text, wait=False):
    """
    Legenerálja a magyar hangot a Google TTS-sel, és lejátssza egy háttérszálon.
    """
    global is_speaking

    def run_speech():
        global is_speaking
        try:
            is_speaking = True

            tts = gTTS(text=text, lang='hu')
            filename = "temp_voice.mp3"
            tts.save(filename)

            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.05)

            pygame.mixer.music.unload()
            try:
                os.remove(filename)
            except:
                pass

        except Exception as e:
            print(f"Hiba a gTTS beszednel: {e}")
        finally:
            time.sleep(0.8)
            is_speaking = False

    if wait:
        run_speech()
    else:
        threading.Thread(target=run_speech).start()


def start_audio_thread():
    """
    listen_in_background-ot használ, hogy soha ne legyen vak folt a figyelésben.
    A callback minden felismert mondat után azonnal lefut.
    """
    global trigger_training_name, trigger_llm_command

    recogniser = sr.Recognizer()
    mic = sr.Microphone()

    # Zajszint kalibrálás induláskor
    print("-> Mikrofon kalibrálása...")
    with mic as source:
        recogniser.adjust_for_ambient_noise(source=source, duration=1.5)
    print("-> Kalibrálás kész.")

    def callback(recogniser, audio):
        global trigger_training_name, trigger_llm_command, is_speaking

        # Ha Zeusz éppen beszél, ne hallja meg a saját hangját
        if is_speaking:
            return

        try:
            text = recogniser.recognize_google(audio, language="hu-HU").lower()

            # Csak Zeusz megszólítására reagálunk
            if "zeusz" not in text:

                return
            print(f"[MIC] Hallottam: '{text}'")
            # Névregisztráció parancs
            if "az én nevem" in text or "nevem" in text:
                if "az én nevem" in text:
                    extracted_name = text.split("az én nevem")[-1].strip()
                else:
                    extracted_name = text.split("nevem")[-1].strip()

                extracted_name = extracted_name.replace(
                    ".", "").replace("?", "").strip()

                if extracted_name:
                    print(f"[MIC] Név bekészítve: '{extracted_name}'")
                    trigger_training_name = extracted_name

            else:
                # Minden más Zeusznak szóló mondat parancsként megy az LLM-nek
                print(f"[MIC] Parancs bekészítve: '{text}'")
                trigger_llm_command = text

        except sr.UnknownValueError:
            # Érthetetlen zaj, csendben továbblépünk
            pass
        except sr.RequestError as e:
            print(f"[MIC] Google STT hiba: {e}")
        except Exception as e:
            print(f"[MIC] Ismeretlen hiba: {e}")

    # Folyamatos háttérfigyelés indítása - soha nem áll le, nincsenek vak foltok
    stop_listening = recogniser.listen_in_background(
        mic, callback, phrase_time_limit=8)
    print("-> 'Zeusz' ébresztési szó figyelése elindult a háttérben...")

    # A stop_listening függvényt visszaadhatnánk ha le akarjuk állítani később,
    # de egy végtelen futású alkalmazásnál erre nincs szükség.
