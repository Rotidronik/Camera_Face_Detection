import time
import subprocess
import speech_recognition as sr
import multiprocessing

RTSP_URL = "rtsp://admin:Rotidronik2002@192.168.10.120:554/h264Preview_01_sub"
SAMPLE_RATE = 16000
CHANNELS = 1

trigger_llm_command = None  # Ez csak a főprocessben használt változó


def _audio_process_worker(command_queue: multiprocessing.Queue):
    """
    Ez a függvény egy TELJESEN KÜLÖN Python processben fut.
    Semmi közös memória nincs az OpenCV-vel, nincs libavcodec ütközés.
    """
    recogniser = sr.Recognizer()
    recogniser.energy_threshold = 5
    recogniser.dynamic_energy_threshold = False
    recogniser.pause_threshold = 2.0
    recogniser.phrase_threshold = 0.1

    print("[AUDIO PROCESS] Elindult a külön audio process.")

    while True:
        print("[AUDIO PROCESS] FFmpeg csatlakozás...")
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
            print("[AUDIO PROCESS] FFmpeg stream aktív.")

            class RTSPAudioSource(sr.AudioSource):
                def __init__(self):
                    self.stream = ffmpeg_process.stdout
                    self.CHUNK = 1024
                    self.SAMPLE_RATE = SAMPLE_RATE
                    self.SAMPLE_WIDTH = 2

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass

            with RTSPAudioSource() as source:
                print("[AUDIO PROCESS] Zajszint kalibrálása...")
                recogniser.adjust_for_ambient_noise(source, duration=2)
                print(
                    f"[AUDIO PROCESS] Kalibrálás kész. Threshold: {recogniser.energy_threshold:.0f}")

                while True:
                    if ffmpeg_process.poll() is not None:
                        print("[AUDIO PROCESS] FFmpeg leállt, újracsatlakozás...")
                        break
                    try:
                        audio = recogniser.listen(
                            source, timeout=10, phrase_time_limit=10)
                        text = recogniser.recognize_google(
                            audio, language="hu-HU").lower()

                        if "zeusz" not in text:
                            continue

                        print(f"[AUDIO PROCESS] Zeusz megszólítva: '{text}'")
                        command_queue.put(text)  # Átadjuk a főprocessnek

                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError as e:
                        print(f"[AUDIO PROCESS] Google STT hiba: {e}")
                        time.sleep(2)
                    except Exception as e:
                        print(f"[AUDIO PROCESS] Hiba: {e}")
                        break

        except Exception as e:
            print(f"[AUDIO PROCESS] Stream hiba: {e}")
        finally:
            try:
                ffmpeg_process.kill()
            except:
                pass

        time.sleep(3)


def start_audio_process():
    """
    Elindít egy teljesen külön Python processzt az audio kezeléshez.
    Visszaad egy Queue-t amin keresztül a főprocess megkapja a parancsokat.
    """
    command_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_audio_process_worker,
        args=(command_queue,),
        daemon=True
    )
    p.start()
    print("[AUDIO] Külön audio process elindítva.")
    return command_queue
