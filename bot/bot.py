#!/usr/bin/env python
# pylint: disable=unused-argument, wrong-import-position
# This program is dedicated to the public domain under the CC0 license.

"""
First, a few callback functions are defined. Then, those functions are passed to
the Application and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Example of a bot-user conversation using ConversationHandler.
Send /start to initiate the conversation.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import logging
import os
import time

import requests
from telegram import __version__ as TG_VER

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 1):
    raise RuntimeError(
        f"This example is not compatible with your current PTB version {TG_VER}. To view the "
        f"{TG_VER} version of this example, "
        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
    )
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

API_TOKEN = os.environ["API_TOKEN"]
APP_HOST = os.environ["APP_HOST"]
APP_KEY = os.environ["APP_KEY"]

session = requests.Session()
session.headers.update({"key": APP_KEY})

check = False
while not check:
    try:
        response = session.get(APP_HOST + "/health")
        check = response.status_code == 200
    except Exception as e:
        logger.info("Exception {} - wait for app".format(e))
        time.sleep(20)

CHOOSING_LANG, CHOOSING_KWS, CHOOSING_NEXT = range(3)


def split_list(a_list):
    half = len(a_list) // 2
    return [a_list[:half], a_list[half:]]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start the conversation and ask user for language."""
    reply_keyboard = [["en"], ["fi"], ["sv"]]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    await update.message.reply_text(
        "Hi! My name is Casper, amd I am a CASual PoEtry cReator. \n\nAs a first step, please select the language for "
        "generating your poem. \n\n[Btw, I do not collect any user information. For future research, I store the poem "
        "lines I generate and your selections.]",
        reply_markup=markup,
    )

    return CHOOSING_LANG


async def set_keywords(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Ask user for input keywords."""
    language = update.message.text
    context.user_data["lang"] = language
    await update.message.reply_text(
        "Language set to {}. Give me keywords.".format(language),
    )

    return CHOOSING_KWS


async def gen_first_line(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Generates first line candidates"""
    keywords = update.message.text.strip()
    context.user_data["keywords"] = update.message.text
    response = session.post(
        url=APP_HOST + "/generator/first_line",
        json={"language": context.user_data["lang"], "style": "", "keywords": keywords},
    )
    context.user_data["last_response"] = response.json()
    candidates = [
        candidate["poem_line"]
        for candidate in context.user_data["last_response"]["candidates"]
    ]
    reply_keyboard = split_list([str(i + 1) for i in range(len(candidates))])
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    out = ""
    for i, candidate in enumerate(candidates):
        out += "\n{} {}".format(i + 1, candidate)
    await update.message.reply_text(
        "Select one of the following candidates:\n{}".format(out),
        reply_markup=markup,
    )

    return CHOOSING_NEXT


async def gen_next_line(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Generates next line candidates"""
    # logger.info(context.user_data["poem_state"])
    selection = int(update.message.text)
    poem_state = context.user_data["last_response"]["candidates"][selection - 1][
        "poem_state"
    ]
    context.user_data["poem"] = poem_state
    response = session.post(
        url=APP_HOST + "/generator/next_line",
        json={
            "language": context.user_data["lang"],
            "style": "",
            "poem_state": poem_state,
            "poem_id": context.user_data["last_response"]["poem_id"],
            "line_id": context.user_data["last_response"]["line_id"],
        },
    )
    context.user_data["last_response"] = response.json()
    candidates = [
        candidate["poem_line"]
        for candidate in context.user_data["last_response"]["candidates"]
    ]
    reply_keyboard = split_list([str(i + 1) for i in range(len(candidates))]) + [
        ["Done"]
    ]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    out = ""
    for i, candidate in enumerate(candidates):
        out += "\n{} {}".format(i + 1, candidate)
    await update.message.reply_text(
        "Poem:\n\n{}\n\n".format("\n".join(context.user_data["poem"]))
        + "Select one of the following candidates:\n{}".format(out),
        reply_markup=markup,
    )

    return CHOOSING_NEXT


async def done(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Display the gathered info and end the conversation."""
    user_data = context.user_data

    await update.message.reply_text(
        "Poem:\n\n{}".format("\n".join(context.user_data["poem"])),
        reply_markup=ReplyKeyboardRemove(),
    )

    user_data.clear()
    return ConversationHandler.END


def main() -> None:
    """Run the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(API_TOKEN).build()

    # Add conversation handler with the states CHOOSING, TYPING_CHOICE and TYPING_REPLY
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            CHOOSING_LANG: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, set_keywords)
            ],
            CHOOSING_KWS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, gen_first_line)
            ],
            CHOOSING_NEXT: [
                MessageHandler(
                    filters.TEXT & ~(filters.COMMAND | filters.Regex("^Done$")),
                    gen_next_line,
                )
            ],
        },
        fallbacks=[MessageHandler(filters.Regex("^Done$"), done)],
    )

    application.add_handler(conv_handler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    main()
