from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# Your bot token
BOT_TOKEN = "telegram_bot_api_token"

# Command to display buttons
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a message with inline buttons."""
    # Create buttons
    keyboard = [
        [InlineKeyboardButton("Known", callback_data="known"), InlineKeyboardButton("Unknown", callback_data="unknown")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Send the message with buttons
    await update.message.reply_text("Intruder detected. Is this person known or unknown?", reply_markup=reply_markup)

# Callback to handle button responses
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles button clicks."""
    query = update.callback_query
    await query.answer()  # Acknowledge the button click

    # Handle user response
    user_response = query.data  # 'known' or 'unknown'
    if user_response == "known":
        await query.edit_message_text(text="You marked this person as Known.")
    elif user_response == "unknown":
        await query.edit_message_text(text="You marked this person as Unknown.")

# Main function to set up the bot
def main():
    # Initialize the bot application
    application = Application.builder().token(BOT_TOKEN).build()

    # Add command and callback query handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button_handler))

    # Start the bot
    application.run_polling()

if __name__ == "__main__":
    main()
