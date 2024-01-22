# SearchSummary
LOCAL ENVIRONMENT

python api.py
OR
uvicorn api:app --host 0.0.0.0 --workers 1 --port 9702 --ssl-keyfile=certs/c5ailabs.com.key --ssl-certfile=certs/c5ailabs.com.crt

docker build -t discovery_anomaly_model_dev .


# for Development

docker run -it -d --restart=always -p 9702:9704 -v /home/shubham/Downloads/Anomaly_detection/Disc_anomaly_detection:/opt --name=discovery_anomaly_model_dev 00a395218d12

# for Production 

docker run -it -d --restart=always -p 9702:9704 -v /home/ubuntu/ai_disc_anomaly/Disc_anomaly_detection:/opt --name=discovery_anomaly_model e4c6238a38d9

# check container logs
docker logs --since=1h 61e843205b16
