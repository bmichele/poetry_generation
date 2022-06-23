FROM python:3.7

#ADD . /code
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./models /code/models
RUN cd /code/models && bash get_models.sh && cd -

COPY ./poem_generator /code/poem_generator
COPY ./app.py /code/app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
