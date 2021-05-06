#!/usr/bin/env python
# coding: utf-8


import nltk
import random
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer



BOT_CONFIG = {'intents': {'hello': {'examples': ['wie es funktioniert', 'Hallo', 'Guten Tag', 'Ich begrüße dich', 'Morgen', 'Halo-halo', 'Guten Morgen!', 'Guten Tag!', 'Guten Abend!', 'Hallöchen', 'öwas', 'Einen schönen guten Tag', 'Glück auf!', 'Ich begrüße Sie', 'Ich heiße Sie herzlich willkommen', 'Herzlich willkommen!', 'Hello', 'Servus veraltend', 'Tagchen', 'MoinMoin', 'Moin', 'GrussGott', 'Hi', 'Gruss dich', 'Daach'], 'responses': [ 'Hallo', 'Guten Tag', 'Ich begrüße dich', 'Morgen', 'Halo-halo', 'Guten Morgen!', 'Guten Tag!', 'Guten Abend!', 'Hallöchen', 'öwas', 'Einen schönen guten Tag!', 'Glück auf!', 'Ich begrüße Sie im Namen Deutscher Freund', 'Ich heiße Sie herzlich willkommen', 'Herzlich willkommen!', 'Hello', 'Servus! veraltend', 'Tagchen!', 'MoinMoin', 'Moin', 'GrussGott', 'Hi', 'Gruss dich', 'Daach!']}, 'bye': {'examples': ['bye', 'Wiedersehen!', 'Auf Wiedersehen!', 'leben Sie wohl!', 'gestatten Sie, dass ich mich verabschiede', 'screibe später', 'Ich empfehle mich!', 'Ich möchte mich empfehlen!', 'Auf ein baldiges Wiedersehen!', 'Auf ein glückliches Wiedersehen!', 'Auf Wiederhören', 'Schönen Abend noch!', 'Wir sehen uns noch!', 'Ich sehe Sie noch!', 'Ich sehe dich (ja) noch!', 'Tschüss', 'macht’s gut', 'Adieu', 'ade', 'ciao', 'Tschau', 'Good bye', 'bay' ], 'responses': ['bye', 'Wiedersehen!', 'Auf Wiedersehen!', 'leben Sie wohl!', 'gestatten Sie, dass ich mich verabschiede', 'screibe später', 'Ich empfehle mich!', 'Ich möchte mich empfehlen!', 'Auf ein baldiges Wiedersehen!', 'Auf ein glückliches Wiedersehen!', 'Auf Wiederhören', 'Schönen Abend noch!', 'Wir sehen uns noch!', 'Ich sehe Sie noch!', 'Ich sehe dich (ja) noch!', 'Tschüss', 'macht’s gut', 'Adieu', 'ade', 'ciao', 'Tschau']}}}


texts = []
intent_names = []

for intent, intent_data in BOT_CONFIG['intents'].items():
    for example in intent_data['examples']:
        texts.append(example)
        intent_names.append(intent)



vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char')
X = vectorizer.fit_transform(texts)
clf = LinearSVC()
clf.fit(X, intent_names)



def classify_intent(replica):
    intent = clf.predict(vectorizer.transform([replica]))[0]

    examples = BOT_CONFIG['intents'][intent]['examples']
    for example in examples:
        example = clear_text(example)
        if len(example) > 0:
            if abs(len(example) - len(replica)) / len(example) < 0.5:
                distance = nltk.edit_distance(replica, example)
                if len(example) and distance / len(example) < 0.5:
                    return intent




def get_answer_by_intent(intent):
    if intent in BOT_CONFIG['intents']:
        responses = BOT_CONFIG['intents'][intent]['responses']
        return random.choice(responses)


# #### generative model

with open('textresult.txt', encoding="utf8") as dialogues_file:
    dialogues_text = dialogues_file.read()
dialogues = dialogues_text.split('\n\n')



def clear_text(text):
    text = text.lower()
    # text = ''.join(char for char in text if char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя -') русский язык
    text = ''.join(char for char in text if char in 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz., -')
    return text


dataset = []  # [[question, answer], ...]
questions = set()

for dialogue in dialogues:
    replicas = dialogue.split('\n')
    replicas = replicas[:2]

    if len(replicas) == 2:
        question, answer = replicas
        question = clear_text(question[2:])
        answer = answer[2:]

        if len(question) > 0 and question not in questions:
            questions.add(question)
            dataset.append([question, answer])


dataset_by_word = {}  # {word: [[question with word, answer], ...], ...}

for question, answer in dataset:
    words = question.split(' ')
    for word in words:
        if word not in dataset_by_word:
            dataset_by_word[word] = []
        dataset_by_word[word].append([question, answer])

dataset_by_word_filtered = {}
for word, word_dataset in dataset_by_word.items():
    word_dataset.sort(key=lambda pair: len(pair[0]))
    dataset_by_word_filtered[word] = word_dataset[:1000]


def generate_answer(replica):
    replica = clear_text(replica)
    if not replica:
        return

    words = set(replica.split(' '))
    words_dataset = []
    for word in words:
        if word in dataset_by_word_filtered:
            word_dataset = dataset_by_word_filtered[word]
            words_dataset += word_dataset

    results = []  # [[question, answer, distance], ...]
    for question, answer in words_dataset:
        if abs(len(question) - len(replica)) / len(question) < 0.2:
            distance = nltk.edit_distance(replica, question)
            if distance / len(question) < 0.2:
                results.append([question, answer, distance])

    question, answer, distance = min(results, key=lambda three: three[2])
    return answer


# #### stubs

def get_stub():
    failure_phrases = BOT_CONFIG['failure_phrases']
    return random.choice(failure_phrases)


# #### bot logic

stats = {'intents': 0, 'generative': 0, 'stubs': 0}


def bot(replica):
    # NLU
    intent = classify_intent(replica)

    # Получение ответа

    # правила
    if intent:
        answer = get_answer_by_intent(intent)
        if answer:
            stats['intents'] += 1
            return answer

    # генеративная модель
    answer = generate_answer(replica)
    if answer:
        stats['generative'] += 1
        return answer

    # заглушка
    answer = get_stub()
    stats['stubs'] += 1
    return answer


# stats

# #### Telegram bot

# get_ipython().system(' pip install python-telegram-bot')

from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext


def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi!')


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def run_bot(update: Update, context: CallbackContext) -> None:
    response = bot(update.message.text)
    update.message.reply_text(response)
    print(update.message.text)
    print(response)
    print(stats)
    print()


def main():
    """Start the bot."""
    updater = Updater("TELEGRAMTOKEN", use_context=True)

    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, run_bot))

    updater.start_polling()
    updater.idle()


main()
