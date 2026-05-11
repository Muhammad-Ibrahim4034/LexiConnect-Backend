FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port (Railway uses $PORT at runtime)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]