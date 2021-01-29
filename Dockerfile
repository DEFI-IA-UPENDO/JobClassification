FROM python:3.7.9

WORKDIR /job-classification

COPY requirements.txt .

RUN pip install -r requirements.txt --ignore-installed

COPY ./app ./app

CMD ["python", "./app/main.py"]