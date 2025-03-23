FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8082

# Run the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8082", "app:app"] 