FROM python:3.8-slim-buster

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app
ADD . /app
RUN pip install --default-timeout=1200 -r requirements.txt

EXPOSE 5000

CMD ["python", "webapp.py", "--port=5000"]