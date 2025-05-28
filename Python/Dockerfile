FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["streamlit", "run", "Test.py", "--server.port=8080", "--server.enableCORS=false"]