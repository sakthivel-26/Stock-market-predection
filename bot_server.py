import json
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

BOT_TOKEN = "YOUR_BOT_TOKEN"

USERS_FILE = "users.json"

def save_user(chat_id):
    try:
        with open(USERS_FILE, "r") as f:
            users = json.load(f)
    except:
        users = []

    if chat_id not in users:
        users.append(chat_id)

    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    save_user(chat_id)
    await update.message.reply_text("✅ You will receive daily stock alerts at 10 AM.")

app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))

app.run_polling()