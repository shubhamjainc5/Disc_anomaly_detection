FROM python:3.8-slim

RUN apt-get update && apt-get -y install \
    build-essential libpq-dev wget

WORKDIR /opt
COPY . .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 9704
ENV PORT 9704

ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "9704", "--ssl-keyfile", "certs/c5ailabs.com.key" , "--ssl-certfile", "certs/c5ailabs.com.crt", "--ssl-ca-certs", "certs/gd_bundle-g2-g1.crt"]
