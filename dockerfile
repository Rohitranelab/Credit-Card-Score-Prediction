FROM python:3.11.3-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

ENV streamlit_app app.py

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]