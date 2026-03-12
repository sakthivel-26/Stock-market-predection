FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860
EXPOSE 8000

CMD uvicorn api:app --host 0.0.0.0 --port 8000 & \
    streamlit run app.py --server.port 7860 --server.address 0.0.0.0