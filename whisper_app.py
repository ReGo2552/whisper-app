from collections import deque
import logging
import os
from pathlib import Path
import sysconfig
import threading
import time
import customtkinter as ctk
import keyboard
import numpy as np
import pyaudiowpatch as pyaudio
import pyperclip
import soxr
from faster_whisper import WhisperModel
from faster_whisper.utils import download_model
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from tokenizers import Tokenizer

HOTKEY = "f8"
MAX_PRE_SECONDS = 30
DEFAULT_PRE_SECONDS = 5
PREVIEW_WINDOW_SECONDS = 30  # Окно для живого предпросмотра (чтобы не тормозило на длинных записях)
FRAMES_PER_BUFFER = 1024

MODEL_NAME = "bzikst/faster-whisper-podlodka-turbo"
TOKENIZER_REPO = "bond005/whisper-podlodka-turbo"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

logger = logging.getLogger(__name__)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class WhisperApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Whisper Podlodka — Буфер")
        self.geometry("460x450")
        self.minsize(380, 380)

        self.model = None
        self.hotkey_handle = None
        self.pa = pyaudio.PyAudio()
        self.stream = None

        self.audio_lock = threading.Lock()
        self.model_lock = threading.Lock()

        self.recording = False
        self.processing = False
        self.session_id = 0
        self.pre_seconds = DEFAULT_PRE_SECONDS

        self.pre_buffer = deque(maxlen=2000)
        self.audio_chunks = []
        self.src_rate = 48000
        self.src_channels = 2

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self._start_audio_stream()
        threading.Thread(target=self._load_model, daemon=True).start()

    def _build_ui(self):
        # Верхняя панель: статус
        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.pack(fill="x", padx=16, pady=(14, 4))

        self.status_label = ctk.CTkLabel(
            top_frame, text="⏳ Загрузка Podlodka Turbo...", font=("Segoe UI", 13, "bold"),
            wraplength=340, justify="left",
        )
        self.status_label.pack(side="left")

        # Строка переключателей: Живой текст + Поверх окон
        opts_frame = ctk.CTkFrame(self, fg_color="transparent")
        opts_frame.pack(fill="x", padx=16, pady=(2, 6))

        self.live_var = ctk.BooleanVar(value=True)
        self.live_checkbox = ctk.CTkCheckBox(
            opts_frame,
            text="Живой текст при записи",
            variable=self.live_var,
            width=20,
            checkbox_width=18,
            checkbox_height=18,
        )
        self.live_checkbox.pack(side="left")

        self.topmost_var = ctk.BooleanVar(value=True)
        self.attributes("-topmost", True)
        ctk.CTkCheckBox(
            opts_frame,
            text="Поверх окон",
            variable=self.topmost_var,
            command=self._toggle_topmost,
            width=20,
            checkbox_width=18,
            checkbox_height=18,
        ).pack(side="right")

        # Главная кнопка записи
        self.rec_btn = ctk.CTkButton(
            self,
            text=f"● Начать запись ({HOTKEY.upper()})",
            height=48,
            font=("Segoe UI", 15, "bold"),
            state="disabled",
            command=self.toggle_recording,
        )
        self.rec_btn.pack(fill="x", padx=16, pady=(4, 10))

        # Настройка захвата ДО начала записи
        pre_frame = ctk.CTkFrame(self, fg_color="#242424", corner_radius=8)
        pre_frame.pack(fill="x", padx=16, pady=(0, 10))

        self.pre_label = ctk.CTkLabel(
            pre_frame,
            text=f"Захват до старта: {self.pre_seconds} сек",
            font=("Segoe UI", 12),
        )
        self.pre_label.pack(anchor="w", padx=12, pady=(8, 0))

        self.pre_slider = ctk.CTkSlider(
            pre_frame,
            from_=0,
            to=MAX_PRE_SECONDS,
            number_of_steps=MAX_PRE_SECONDS,
            command=self._on_pre_slider_change,
        )
        self.pre_slider.set(self.pre_seconds)
        self.pre_slider.pack(fill="x", padx=12, pady=(4, 10))

        # Поле вывода текста
        self.textbox = ctk.CTkTextbox(self, font=("Segoe UI", 13), wrap="word")
        self.textbox.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        # Нижняя панель
        bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        bottom_frame.pack(fill="x", padx=16, pady=(0, 14))

        ctk.CTkButton(
            bottom_frame,
            text="Скопировать",
            width=115,
            fg_color="#333333",
            hover_color="#444444",
            command=self._copy_textbox,
        ).pack(side="left")

        ctk.CTkButton(
            bottom_frame,
            text="Обновить звук",
            width=115,
            fg_color="#333333",
            hover_color="#444444",
            command=self._restart_audio_stream,
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            bottom_frame,
            text="Очистить",
            width=85,
            fg_color="#333333",
            hover_color="#444444",
            command=lambda: self.textbox.delete("1.0", "end"),
        ).pack(side="right")

    def _toggle_topmost(self):
        self.attributes("-topmost", self.topmost_var.get())

    def _on_pre_slider_change(self, value: float):
        self.pre_seconds = int(round(value))
        self.pre_label.configure(text=f"Захват до старта: {self.pre_seconds} сек")

    def _copy_textbox(self):
        content = self.textbox.get("1.0", "end").strip()
        if content:
            pyperclip.copy(content)
            self.status_label.configure(text="✅ Скопировано в буфер", text_color="#4ade80")

    def _set_text(self, text: str, sid: int, is_draft: bool = False):
        """Безопасное обновление поля текста с защитой от устаревших потоков."""
        if sid != self.session_id:
            return
        if is_draft and not self.recording:
            return
        color = "#9ca3af" if is_draft else "#f3f4f6"
        self.textbox.configure(text_color=color)
        self.textbox.delete("1.0", "end")
        self.textbox.insert("1.0", text)
        self.textbox.see("end")

    def _load_model(self):
        try:
            try:
                model_path = Path(download_model(MODEL_NAME, local_files_only=True))
            except LocalEntryNotFoundError:
                model_path = None
            required_files = ("model.bin", "config.json", "vocabulary.json", "preprocessor_config.json")
            if model_path is None or not all((model_path / name).is_file() for name in required_files):
                model_path = Path(download_model(MODEL_NAME))

            try:
                tokenizer_path = hf_hub_download(TOKENIZER_REPO, "tokenizer.json", local_files_only=True)
            except LocalEntryNotFoundError:
                tokenizer_path = hf_hub_download(TOKENIZER_REPO, "tokenizer.json")
            tokenizer = Tokenizer.from_file(tokenizer_path)

            is_cuda = os.environ.get("CUDA_VISIBLE_DEVICES") not in ("-1", "")
            if is_cuda:
                self._add_cuda_libraries_to_path()
                self.after(0, lambda: self.status_label.configure(text="⏳ Проверка GPU..."))
                try:
                    self.model = self._open_model(model_path, tokenizer, DEVICE, COMPUTE_TYPE)
                except Exception as exc:
                    logger.warning("GPU недоступен, переключаюсь на CPU: %s", exc)
                    is_cuda = False

            if not is_cuda:
                self.after(0, lambda: self.status_label.configure(text="⏳ Загрузка модели на CPU..."))
                self.model = self._open_model(model_path, tokenizer, "cpu", "int8")
        except Exception as exc:
            logger.exception("Не удалось запустить модель")
            self.after(0, lambda error=str(exc): self.status_label.configure(
                text=f"❌ Ошибка модели: {error}", text_color="#f87171"
            ))
            return

        self.after(0, lambda: self._on_model_ready(is_cuda))

    @staticmethod
    def _add_cuda_libraries_to_path():
        nvidia_dir = Path(sysconfig.get_path("purelib")) / "nvidia"
        dll_dirs = [nvidia_dir / package / "bin" for package in ("cublas", "cudnn", "cuda_nvrtc")]
        paths = [str(path) for path in dll_dirs if path.is_dir()]
        if paths:
            os.environ["PATH"] = os.pathsep.join([*paths, os.environ.get("PATH", "")])

    @staticmethod
    def _open_model(model_path, tokenizer, device, compute_type):
        model = WhisperModel(str(model_path), device=device, compute_type=compute_type)
        model.hf_tokenizer = tokenizer
        if device == "cuda":
            # Библиотеки CUDA могут загрузиться только при первом распознавании.
            segments, _ = model.transcribe(
                np.zeros(16000, dtype=np.float32),
                language="ru",
                beam_size=1,
                vad_filter=False,
                without_timestamps=True,
            )
            list(segments)
        return model

    def _on_model_ready(self, is_cuda: bool):
        mode = "GPU" if is_cuda else "CPU"
        if not is_cuda:
            # На CPU живой предпросмотр лучше отключить по умолчанию (но можно включить галочкой)
            self.live_var.set(False)
        self.rec_btn.configure(state="normal")
        try:
            self.hotkey_handle = keyboard.add_hotkey(
                HOTKEY, lambda: self.after(0, self.toggle_recording), trigger_on_release=True
            )
        except Exception:
            logger.exception("Не удалось зарегистрировать %s", HOTKEY.upper())
            self.status_label.configure(
                text=f"⚠️ Готов ({mode}). {HOTKEY.upper()} недоступна, используй кнопку.",
                text_color="#facc15",
            )
            return
        if self.stream is None:
            self.status_label.configure(
                text="⚠️ Системный звук недоступен. Нажми «Обновить звук».", text_color="#facc15"
            )
        else:
            self.status_label.configure(text=f"🟢 Готов ({mode})", text_color="#4ade80")

    def _get_loopback_device(self):
        return self.pa.get_default_wasapi_loopback()

    def _start_audio_stream(self):
        try:
            dev = self._get_loopback_device()
            self.src_rate = int(dev["defaultSampleRate"])
            self.src_channels = dev["maxInputChannels"]

            max_chunks = int(((MAX_PRE_SECONDS + 5) * self.src_rate) / FRAMES_PER_BUFFER) + 50
            with self.audio_lock:
                self.pre_buffer = deque(maxlen=max_chunks)

            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.src_channels,
                rate=self.src_rate,
                input=True,
                input_device_index=dev["index"],
                frames_per_buffer=FRAMES_PER_BUFFER,
                stream_callback=self._audio_callback,
            )
            return True
        except Exception as e:
            self.status_label.configure(text=f"❌ Ошибка аудио: {e}", text_color="#f87171")
            return False

    def _restart_audio_stream(self):
        if self.recording or self.processing:
            return
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        if self._start_audio_stream() and self.model is not None:
            self.status_label.configure(text="🟢 Устройство звука обновлено", text_color="#4ade80")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        now = time.monotonic()
        with self.audio_lock:
            if self.recording:
                self.audio_chunks.append(in_data)
            else:
                self.pre_buffer.append((now, in_data))
        return (None, pyaudio.paContinue)

    def _bytes_to_16k(self, raw_bytes: bytes, rate: int, channels: int) -> np.ndarray:
        pcm = np.frombuffer(raw_bytes, dtype=np.int16)
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1)
        audio_f32 = pcm.astype(np.float32) / 32768.0
        return soxr.resample(audio_f32, rate, 16000).astype(np.float32)

    def toggle_recording(self):
        if self.model is None or self.processing:
            return

        if not self.recording:
            if self.stream is None or not self.stream.is_active():
                if not self._start_audio_stream():
                    return

            self.session_id += 1
            sid = self.session_id
            now = time.monotonic()
            cutoff = now - self.pre_seconds

            with self.audio_lock:
                if self.pre_seconds > 0:
                    self.audio_chunks = [
                        chunk for ts, chunk in self.pre_buffer if ts >= cutoff
                    ]
                else:
                    self.audio_chunks = []
                self.pre_buffer.clear()
                self.recording = True

            self.textbox.configure(text_color="#9ca3af")
            self.textbox.delete("1.0", "end")
            self.status_label.configure(text="🔴 Идет запись звука...", text_color="#f87171")
            self.rec_btn.configure(
                text=f"■ Остановить ({HOTKEY.upper()})",
                fg_color="#dc2626",
                hover_color="#b91c1c",
            )

            # Запускаем фоновый живой предпросмотр, если включена галочка
            if self.live_var.get():
                threading.Thread(
                    target=self._live_preview_worker,
                    args=(sid, self.src_rate, self.src_channels),
                    daemon=True,
                ).start()
        else:
            sid = self.session_id
            with self.audio_lock:
                self.recording = False
                chunks_copy = list(self.audio_chunks)
                self.audio_chunks = []

            raw_bytes = b"".join(chunks_copy)
            self.processing = True
            self.status_label.configure(text="⏳ Финальное распознавание...", text_color="#facc15")
            self.rec_btn.configure(
                text="Обработка...",
                state="disabled",
                fg_color="#1f6aa5",
                hover_color="#144870",
            )

            threading.Thread(
                target=self._final_transcribe_worker,
                args=(sid, raw_bytes, self.src_rate, self.src_channels),
                daemon=True,
            ).start()

    def _live_preview_worker(self, sid: int, rate: int, channels: int):
        last_chunk_count = 0
        min_bytes = int(rate * channels * 2 * 0.4)  # минимум 0.4 сек звука
        max_window_bytes = int(rate * channels * 2 * PREVIEW_WINDOW_SECONDS)

        while self.recording and sid == self.session_id:
            if not self.live_var.get():
                time.sleep(0.3)
                continue

            with self.audio_lock:
                curr_count = len(self.audio_chunks)
                if curr_count == last_chunk_count:
                    chunks_copy = None
                else:
                    chunks_copy = list(self.audio_chunks)
                    last_chunk_count = curr_count

            if not chunks_copy:
                time.sleep(0.25)
                continue

            raw_bytes = b"".join(chunks_copy)
            if len(raw_bytes) < min_bytes:
                time.sleep(0.25)
                continue

            truncated = False
            if len(raw_bytes) > max_window_bytes:
                raw_bytes = raw_bytes[-max_window_bytes:]
                truncated = True

            t0 = time.monotonic()
            try:
                audio_16k = self._bytes_to_16k(raw_bytes, rate, channels)
                with self.model_lock:
                    if not self.recording or sid != self.session_id:
                        break
                    segments, _ = self.model.transcribe(
                        audio_16k,
                        language="ru",
                        beam_size=1,
                        vad_filter=True,
                        without_timestamps=True,
                        condition_on_previous_text=False,
                    )
                    parts = []
                    for seg in segments:
                        # Мгновенно выходим, если пользователь уже нажал "Стоп"
                        if not self.recording or sid != self.session_id:
                            break
                        piece = seg.text.strip()
                        if piece:
                            parts.append(piece)
                            prefix = "… " if truncated else ""
                            preview_text = prefix + " ".join(parts)
                            self.after(
                                0,
                                lambda txt=preview_text: self._set_text(txt, sid, is_draft=True),
                            )
            except Exception:
                pass

            # Адаптивная пауза: не нагружаем GPU/CPU чаще, чем раз в 0.7 сек
            elapsed = time.monotonic() - t0
            sleep_time = max(0.7, elapsed * 0.8)
            deadline = time.monotonic() + sleep_time
            while self.recording and sid == self.session_id and time.monotonic() < deadline:
                time.sleep(0.1)

    def _final_transcribe_worker(self, sid: int, raw_bytes: bytes, rate: int, channels: int):
        if len(raw_bytes) < int(rate * channels * 2 * 0.1):
            self.after(0, lambda: self._finish_transcription(sid, "", "⚠️ Звук не записался (тишина)"))
            return

        try:
            audio_16k = self._bytes_to_16k(raw_bytes, rate, channels)
            parts = []

            with self.model_lock:
                segments, _ = self.model.transcribe(
                    audio_16k,
                    language="ru",
                    beam_size=1,
                    vad_filter=True,
                    without_timestamps=True,
                    condition_on_previous_text=False,
                )
                # Выводим каждый сегмент в окно сразу по мере распознавания
                for seg in segments:
                    piece = seg.text.strip()
                    if piece:
                        parts.append(piece)
                        current_text = " ".join(parts)
                        self.after(
                            0,
                            lambda txt=current_text: self._set_text(txt, sid, is_draft=False),
                        )

            full_text = " ".join(parts).strip()
            if full_text:
                pyperclip.copy(full_text)
                self.after(
                    0,
                    lambda: self._finish_transcription(sid, full_text, "✅ Скопировано в буфер (Ctrl+V)"),
                )
            else:
                self.after(0, lambda: self._finish_transcription(sid, "", "⚠️ Речь не обнаружена"))
        except Exception as e:
            self.after(0, lambda err=e: self._finish_transcription(sid, "", f"❌ Ошибка: {err}"))

    def _finish_transcription(self, sid: int, text: str, status: str):
        if sid != self.session_id:
            return
        self.processing = False
        if text:
            self._set_text(text, sid, is_draft=False)

        color = "#4ade80" if status.startswith("✅") else "#facc15"
        self.status_label.configure(text=status, text_color=color)
        self.rec_btn.configure(
            text=f"● Начать запись ({HOTKEY.upper()})",
            state="normal",
        )

    def on_close(self):
        self.recording = False
        if self.hotkey_handle is not None:
            keyboard.remove_hotkey(self.hotkey_handle)
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
        self.pa.terminate()
        self.destroy()


if __name__ == "__main__":
    app = WhisperApp()
    app.mainloop()
