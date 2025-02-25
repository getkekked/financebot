#!/usr/bin/env python
# coding: utf-8


import yfinance as yf
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from langchain.prompts import PromptTemplate
import re
import requests
import google.generativeai as genai
from langchain.chains import SequentialChain, LLMChain, RouterChain
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

GNEWS_API_TOKEN = os.getenv("GNEWS_API_TOKEN")
GEMINI_API_TOKEN = os.getenv("GEMINI_API_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
X_API_KEY =  os.getenv("X_API_KEY")


nltk.download("stopwords")
nltk.download("punkt")


def preprocess_query(query):
    """Очищает запрос от стоп-слов и возвращает ключевые слова."""
    stop_words = set(stopwords.words("russian"))
    query = query.lower()
    query = query.replace(";", " OR ")
    query = re.sub(r"[^а-яА-Яa-zA-Z0-9 ]", "", query)
    words = word_tokenize(query)
    return " ".join(words)

def get_gnews(query):
    """Запрашивает новости из GNews API по ключевым словам."""
    processed_query = preprocess_query(query)
    url = f"https://gnews.io/api/v4/search?q={processed_query}&lang=ru&country=ru&category=business&sortby=publishedAt&token={GNEWS_API_TOKEN}"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return [{"title": item["title"], "url": item["url"]} for item in data.get("articles", [])[:3]]
    else:
        return ["Ошибка при получении новостей"]

def summarize_news(news_articles, llm):
    """Генерирует краткое содержание статьи и возвращает заголовок, пересказ и ссылку."""
    summaries = []
    
    for article in news_articles:
        title = article["title"]
        url = article["url"]
        content = extract_article_text(url)
        
        if not content or "Ошибка" in content:
            summary = "Не удалось получить текст статьи."
        else:
            try:
                prompt = f"""
                Прочитай следующий текст статьи (урезан до 4000 символов) и сделай краткий пересказ в 2-3 предложениях.
                Если какой-то информации в статье нет, не выдумывай ее, очень важно, чтобы пересказ был точным и
                не вводил пользователя в заблуждение ложной информацией. Текст статьи:
                "{content[:4000]}"
                """
                response = llm(prompt)
                summary = response
            except ValueError as e:
                summary = "Ответ заблокирован системой безопасности."

        summaries.append(f"📰 {title}\n\n📌 {summary}\n\n🔗 {url}")
    
    return "\n\n".join(summaries)


from bs4 import BeautifulSoup

def extract_article_text(url):
    """Извлекает текст статьи с помощью BeautifulSoup."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return f"Ошибка: {response.status_code}"

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text() for p in paragraphs)

        return text if text else "Ошибка: текст не найден"
    
    except Exception as e:
        return f"Ошибка при извлечении статьи: {str(e)}"

def get_news_summary(query, llm):
    news_articles = get_gnews(query)
    return summarize_news(news_articles, llm)

def get_stock_price(search_term):
    results = []
    query = requests.get(f'https://yfapi.net/v6/finance/autocomplete?region=IN&lang=en&query={search_term}', 
    headers={
        'accept': 'application/json',
        'X-API-KEY': X_API_KEY
    })
    response = query.json()
    for i in response['ResultSet']['Result']:
        final = i['symbol']
        results.append(final)

    if not results:
        return None
    stock = yf.Ticker(results[0])
    history = stock.history(period="1d")
    latest_data = history.iloc[-1]
    stock_info = {
        "symbol": results[0],
        "date": latest_data.name.strftime("%Y-%m-%d"),
        "open": latest_data["Open"],
        "high": latest_data["High"],
        "low": latest_data["Low"],
        "close": latest_data["Close"],
        "volume": latest_data["Volume"]
    }
    return stock_info

def get_stock_summary(query, llm):
    company_name = extract_company_name(query, llm)
    if not company_name:
        return "Компания не найдена."

    stock_info = get_stock_price(company_name)
    if stock_info is None:
        return "Не удалось найти котировки акций этой компании."
    
    stock_summary = f"""
    Данные на {stock_info['date']}:
    Акции {stock_info['symbol']}:
    - Открытие: {stock_info['open']:.2f} USD
    - Максимум: {stock_info['high']:.2f} USD
    - Минимум: {stock_info['low']:.2f} USD
    - Закрытие: {stock_info['close']:.2f} USD
    - Объем торгов: {stock_info['volume']}
    """

    response = llm(f"{stock_summary}\n\nДай краткий обзор ситуации с котировками этих акций в двух-трех предложениях ")
    
    return stock_summary + "\n\n" + response

def extract_company_name(query, llm):
    """Извлекает название компании из запроса с помощью Gemini."""
    prompt = f"""
    Выдели название компании, если оно есть, и переведи его на латиницу, если оно написано кириллицей, из следующего запроса:
    "{query}"
    Ответь только названием компании без лишнего текста.
    """
    response = llm(prompt)
    company_name = response.strip()

    return company_name

def extract_entity(query, llm):
    """Извлекает название компании из запроса с помощью Gemini."""
    prompt = f"""
    Выдели сущность или несколько сущностей, которых касается запрос пользователя, из следующего запроса (Пример: "Как дела с выборами в США?" -> "выборы США"):
    "{query}"
    Назови только эти сущности без лишнего текста и знаков. Исправь опечатки там, где считаешь нужным. Если сущностей больше чем одна, раздели из с помощью знака ;
    """
    response = llm(prompt)
    entity = response.strip()

    return entity


genai.configure(api_key=GEMINI_API_TOKEN)

class CustomGeminiLLM:
    """Обёртка для Gemini, совместимая с LangChain."""
    def __init__(self, model_name="gemini-pro", temperature=0.7):
        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature

    def __call__(self, query):
        """Вызывает модель и возвращает текст."""
        if isinstance(query, str):
            prompt = query
        else:
            prompt = query.text 
        response = self.model.generate_content(prompt)
        return response.text.strip()


category_prompt_template = """
Тебя зовут Сайга, ты - русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им быть в курсе новостей финансового рынка.
Определи, что запрос касается: новостей или акций какой-либо компании;
какого-то объекта или события, о котором могут быть новости (Пример: Выборы в США, Мексиканский залив); или другого. 
Ответь односложно: 'компания', 'общее' или 'другое'.
Пожалуйста, не добавляй ничего лишнего, только одно слово.
Запрос: {input}
"""

basic_template = """
Тебя зовут Сайга, ты - русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им быть в курсе новостей финансового рынка.
Старайся отвечать кратко, от одного до трех предложений.
Запрос: {query}
"""

company_prompt = PromptTemplate.from_template("Расскажи про ситуацию с котировками компании на основе этих данных в 2-3 предложениях: {stock_info}.")

category_prompt = PromptTemplate(input_variables=["input"], template=category_prompt_template)

basic_prompt = PromptTemplate(input_variables=["query"], template=basic_template)

model = CustomGeminiLLM()

router_chain = (
    {"query": RunnablePassthrough()}
    | {
        "query": lambda x: x, 
        "category": category_prompt | model
    }
    | RunnableBranch(
        (RunnableLambda(lambda x: x["category"] == "компания"), RunnableLambda(lambda x: process_company_query(x["query"]))),
        (RunnableLambda(lambda x: x["category"] == "общее"), RunnableLambda(lambda x: process_general_news(x["query"]))),
        RunnableLambda(lambda x: basic_prompt | model)
    )
)

def process_company_query(query):
    company_name = extract_company_name(query, model)
    if not company_name:
        return "Компания не найдена."
    return get_stock_summary(company_name, model) + '\n\n' + get_news_summary(company_name, model)

def process_general_news(query):
    entity = extract_entity(query, model)
    return get_news_summary(entity, model)
                

async def start(update: Update, context: CallbackContext) -> None:
    """Отправляет приветственное сообщение при старте бота."""
    await update.message.reply_text("Привет! Отправь мне запрос о рынке или акциях.")

async def handle_message(update: Update, context: CallbackContext) -> None:
    """Обрабатывает сообщение пользователя."""
    user_query = update.message.text
    response = router_chain.invoke(user_query)
    await update.message.reply_text(response)

def main():
    """Запускает Telegram-бота."""
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Бот запущен!")
    app.run_polling()

if __name__ == "__main__":
    main()

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    response = router_chain.invoke(message.text)
    bot.send_message(message.chat.id, response)

print("Бот запущен...")
bot.polling()




