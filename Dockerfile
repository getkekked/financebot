FROM python:3.12.4

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /usr/share/nltk_data \
    && python -c "import nltk; nltk.data.path.append('/usr/share/nltk_data'); nltk.download('punkt', download_dir='/usr/share/nltk_data')"

ENV NLTK_DATA=/usr/share/nltk_data

ENV PYTHONUNBUFFERED=1

CMD ["python", "chatbot_gemini.py"]