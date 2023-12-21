# SearchSummary

uvicorn app:app --host 0.0.0.0 --workers 1 --port 9702

docker build -t discovery_anomaly_model .


for Development

docker run -it -d --restart=always -p 9702:9702 -v /home/shubham/Downloads/Anomaly_detection/custom_anamoly:/opt --name=discovery_anomaly_model 00a395218d12

for Production 

docker run -it -d --restart=always -p 9050:9050 -v /home/ubuntu/summary_qa/SearchSummary/webapp:/opt --name=summary_qa eec0a00fb58b