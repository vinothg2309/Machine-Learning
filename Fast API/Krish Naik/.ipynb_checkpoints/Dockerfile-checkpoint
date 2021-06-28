FROM python:3.8-slim
COPY . /usr/app/
EXPOSE 5001
WORKDIR /usr/app/
RUN pip install -r requirements.txt
CMD python main.py