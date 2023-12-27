# SearchSummary

python api.py
OR
uvicorn app:app --host 0.0.0.0 --workers 1 --port 9702

docker build -t discovery_anomaly_model .


for Development

docker run -it -d --restart=always -p 9702:9702 -v /home/shubham/Downloads/Anomaly_detection/Disc_anomaly_detection:/opt --name=discovery_anomaly_model 00a395218d12

for Production 

docker run -it -d --restart=always -p 9702:9702 -v /home/ubuntu/ai_disc_anomaly/Disc_anomaly_detection:/opt --name=discovery_anomaly_model e4c6238a38d9