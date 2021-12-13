FROM public.ecr.aws/lambda/python:3.8

COPY requirements.txt ./
COPY HDFS_2k.csv ./
COPY HDFS_100k.csv ./
COPY anomaly_label.csv ./
RUN pip3 install -r requirements.txt
COPY app.py ./

CMD ["app.lambda_handler"]
