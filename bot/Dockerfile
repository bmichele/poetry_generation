FROM python:3.7

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./bot.py /code/bot.py

CMD ["python", "bot.py"]
