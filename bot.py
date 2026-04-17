"""
Telegram bot for audio transcription via Yandex SpeechKit.
Handles voice messages and audio file uploads.
"""

import io
import logging
import os
import tempfile

import google.generativeai as genai
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

from transcribator import transcribe, get_duration, find_ffmpeg

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MAX_FILE_MB = 20  # Telegram Bot API limit for downloads
TEXT_AS_FILE_CHARS = 1000  # send as .txt file if longer


def format_text(text: str) -> str:
    """Add punctuation and paragraph breaks using Gemini."""
    if not GEMINI_API_KEY:
        return text
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = (
        "Ты редактор текста. Твоя задача — расставить знаки препинания (точки, запятые, вопросительные и восклицательные знаки) "
        "и разбить текст на абзацы по смыслу. "
        "Не меняй ни одного слова, не добавляй и не убирай слова. "
        "Верни только отредактированный текст, без пояснений.\n\n"
        "Текст:\n" + text
    )
    response = model.generate_content(prompt)
    return response.text


def seconds_to_human(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} сек"
    m, s = divmod(seconds, 60)
    return f"{m} мин {s} сек" if s else f"{m} мин"


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я транскрибирую аудио в текст.\n\n"
        "Просто отправь голосовое сообщение или аудиофайл.\n\n"
        "Поддерживаемые форматы: m4a, mp3, ogg, wav, opus, flac и другие.\n"
        "Максимальный размер файла: 50 МБ.\n\n"
        "Язык определяется автоматически (русский, английский и др.).\n\n"
        "/help — помощь"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Как пользоваться:\n\n"
        "1. Отправь голосовое сообщение или аудиофайл\n"
        "2. Дождись транскрибации\n"
        "3. Получи текст\n\n"
        "Если текст длинный — он придёт как файл .txt, удобный для копирования.\n\n"
        "Поддерживаемые форматы:\n"
        "m4a, mp3, ogg, opus, wav, flac, wma и другие\n\n"
        "Ограничения:\n"
        f"— Максимальный размер файла: {MAX_FILE_MB} МБ\n"
        "— Язык определяется автоматически\n\n"
        "По вопросам: @sagyndykovmax"
    )


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message

    if message.voice:
        tg_file_obj = message.voice
        ext = ".ogg"
        file_size = tg_file_obj.file_size
    elif message.audio:
        tg_file_obj = message.audio
        name = tg_file_obj.file_name or "audio.mp3"
        ext = os.path.splitext(name)[1] or ".mp3"
        file_size = tg_file_obj.file_size
    else:
        return

    # Check file size
    if file_size and file_size > MAX_FILE_MB * 1024 * 1024:
        size_mb = file_size / 1024 / 1024
        await message.reply_text(
            f"Файл слишком большой ({size_mb:.1f} МБ). Максимум — {MAX_FILE_MB} МБ.\n\n"
            "Как уменьшить файл:\n"
            "1. На iPhone: запись → поделиться → сохранить через голосовые сообщения\n"
            "2. Разбейте запись на части по 15-20 минут\n"
            "3. Используйте приложение Voice Recorder вместо стандартного"
        )
        return

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = os.path.join(tmp, f"audio{ext}")
        tg_file = await tg_file_obj.get_file()
        await tg_file.download_to_drive(audio_path)

        # Get duration and inform user
        try:
            ffmpeg = find_ffmpeg()
            duration = get_duration(ffmpeg, audio_path)
            duration_str = seconds_to_human(duration)
            # ~1.5s per 25s chunk + 2s conversion overhead
            chunks = max(1, int(duration / 25) + 1)
            est_seconds = int(chunks * 1.5 + 2)
            if est_seconds < 10:
                est_str = "несколько секунд"
            else:
                est_str = f"~{seconds_to_human(est_seconds)}"
            await message.reply_text(
                f"Длительность: {duration_str}\n"
                f"Транскрибирую, это займёт {est_str}..."
            )
        except Exception:
            await message.reply_text("Транскрибирую...")

        try:
            text = transcribe(audio_path, lang="auto", api_key=YANDEX_API_KEY, verbose=False)
        except Exception as e:
            logger.error("Transcription error: %s", e)
            await message.reply_text(
                "Не удалось обработать аудио. "
                "Убедитесь что файл не повреждён и попробуйте ещё раз."
            )
            return

    if not text.strip():
        await message.reply_text("Не удалось распознать речь в этом аудио.")
        return

    if GEMINI_API_KEY:
        try:
            text = format_text(text)
        except Exception as e:
            logger.error("Formatting error: %s", e)

    # Send as file if text is long, otherwise as message
    if len(text) > TEXT_AS_FILE_CHARS:
        file_bytes = io.BytesIO(text.encode("utf-8"))
        file_bytes.name = "transcription.txt"
        await message.reply_document(
            document=file_bytes,
            filename="transcription.txt",
            caption=f"Транскрипция готова ({len(text)} символов)"
        )
    else:
        await message.reply_text(text)


def main() -> None:
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set in .env")
    if not YANDEX_API_KEY:
        raise RuntimeError("YANDEX_API_KEY not set in .env")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))

    logger.info("Bot started")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
