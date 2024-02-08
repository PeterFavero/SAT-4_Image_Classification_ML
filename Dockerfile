FROM python:3.11

RUN mkdir usr/app
WORKDIR usr/app

COPY . .

RUN pip install -r requirements.txt
CMD python3 src/RunAll.py;