"""
Telegram bot for audio transcription via Yandex SpeechKit.
Handles voice messages and audio file uploads.
"""

import logging
import os
import tempfile

import google.generativeai as genai
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

from transcribator import transcribe

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


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


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Отправь голосовое сообщение или аудиофайл — я транскрибирую его в текст."
    )


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message

    # Determine file source: voice or audio
    if message.voice:
        tg_file = await message.voice.get_file()
        ext = ".ogg"
    elif message.audio:
        tg_file = await message.audio.get_file()
        # Try to preserve original extension
        name = message.audio.file_name or "audio.mp3"
        ext = os.path.splitext(name)[1] or ".mp3"
    else:
        return

    await message.reply_text("Транскрибирую...")

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = os.path.join(tmp, f"audio{ext}")
        await tg_file.download_to_drive(audio_path)

        try:
            text = transcribe(audio_path, lang="auto", api_key=YANDEX_API_KEY, verbose=False)
        except Exception as e:
            logger.error("Transcription error: %s", e)
            await message.reply_text(f"Ошибка транскрибации: {e}")
            return

    if not text.strip():
        await message.reply_text("Не удалось распознать речь.")
        return

    if GEMINI_API_KEY:
        await message.reply_text("Форматирую...")
        try:
            text = format_text(text)
        except Exception as e:
            logger.error("Formatting error: %s", e)

    # Telegram message limit is 4096 chars — split if needed
    for i in range(0, len(text), 4096):
        await message.reply_text(text[i:i + 4096])


def main() -> None:
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set in .env")
    if not YANDEX_API_KEY:
        raise RuntimeError("YANDEX_API_KEY not set in .env")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))

    logger.info("Bot started")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
