import time
import subprocess
import speech_recognition as sr
import multiprocessing

RTSP_URL = "rtsp://admin:Rotidronik2002@192.168.10.120:554/h264Preview_01_sub"
SAMPLE_RATE = 16000
CHANNELS = 1


def _audio_process_worker(command_queue: multiprocessing.Queue):
    """Folyamatosan futó audio stream, indításkori egyszeri zajszint-kalibrációval."""
    recogniser = sr.Recognizer()

    # MEGEMELT IDŐZÍTÉSEK: hagyja, hogy kényelmesen, természetesen elmondd a mondatot
    # 1.3 másodperc tiszta csend kell a mondat végének azonosításához
    recogniser.pause_threshold = 1.3
    # Minimum 0.3 másodperces beszéd kell, hogy egyáltalán reagáljon
    recogniser.phrase_threshold = 0.3
    recogniser.non_speaking_duration = 0.6  # A beszéd előtti/utáni puffer

    print("[AUDIO PROCESS] Elindult a permanens audio process.")

    while True:
        try:
            ffmpeg_process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-loglevel", "quiet",
                    "-rtsp_transport", "tcp",
                    "-i", RTSP_URL,
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", str(SAMPLE_RATE),
                    "-ac", str(CHANNELS),
                    "-f", "s16le",
                    "pipe:1"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )

            class RTSPAudioSource(sr.AudioSource):
                def __init__(self):
                    self.stream = ffmpeg_process.stdout
                    self.CHUNK = 1024
                    self.SAMPLE_RATE = SAMPLE_RATE
                    self.SAMPLE_WIDTH = 2

                def __enter__(self): return self
                def __exit__(self, exc_type, exc_val, exc_tb): pass

            with RTSPAudioSource() as source:
                # KRITIKUS LÉPÉS: Egyszeri automatikus zajszint-kalibráció az indítás pillanatában
                print(
                    "[AUDIO PROCESS] Reolink zajpadló automatikus kalibrálása (1 mp)...")
                recogniser.adjust_for_ambient_noise(source, duration=1)
                print(
                    f"[AUDIO PROCESS] Kalibráció kész! Dinamikus küszöbérték beállítva: {recogniser.energy_threshold:.0f}")
                print("[AUDIO PROCESS] Figyelés elindítva...")

                while True:
                    if ffmpeg_process.poll() is not None:
                        break
                    try:
                        # Nem timeoutol le a végtelenbe, de engedi a hosszú mondatokat is
                        audio = recogniser.listen(
                            source, timeout=None, phrase_time_limit=10)
                        text = recogniser.recognize_google(
                            audio, language="hu-HU").lower()

                        if "zeusz" in text:
                            print(
                                f"[AUDIO PROCESS] Zeusz megszólítva: '{text}'")
                            command_queue.put(text)

                    except sr.UnknownValueError:
                        # Ha nem tudta értelmezni a hangot (pl. elment egy autó), nem száll el, megy tovább csendben
                        continue
                    except Exception as e:
                        print(f"[AUDIO PROCESS] Belső hiba a hurokban: {e}")
                        break

        except Exception as e:
            print(f"[AUDIO PROCESS] FFmpeg kapcsolódási hiba: {e}")
        finally:
            try:
                ffmpeg_process.kill()
            except:
                pass

        time.sleep(2)


def start_audio_process():
    command_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_audio_process_worker, args=(command_queue,), daemon=True)
    p.start()
    return command_queue
