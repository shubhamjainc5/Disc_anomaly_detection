# SearchSummary
DEVELOPMENT ENVIRONMENT

python api.py
OR
uvicorn app:app --host 0.0.0.0 --workers 1 --port 9704

docker build -t discovery_anomaly_model_dev .


for Development

docker run -it -d --restart=always -p 9704:9704 -v /home/shubham/Downloads/Anomaly_detection/Disc_anomaly_detection:/opt --name=discovery_anomaly_model_dev 00a395218d12

for Production 

docker run -it -d --restart=always -p 9704:9704 -v /home/ubuntu/ai_disc_anomaly_dev/Disc_anomaly_detection:/opt --name=discovery_anomaly_model_dev e4c6238a38d9
