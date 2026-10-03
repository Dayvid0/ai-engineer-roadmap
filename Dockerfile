FROM python:3.12-slim
WORKDIR /app
COPY requirements_light.txt .
RUN pip install --no-cache-dir -r requirements_light.txt
COPY rag_api_light.py .
COPY my_roadmap_notes.txt .
EXPOSE 8000
CMD ["uvicorn", "rag_api_light:app", "--host", "0.0.0.0", "--port", "8000"]
