FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 9001

ENV NATS_SERVER_URL=nats://nats:4222
ENV NATS_IN_SUBJECT=workflow.previousagent.result
ENV NATS_IN_DURABLE=workflow-previousagent-result
ENV NATS_OUT_SUBJECT=workflow.agenttemplate.result

CMD ["python", "main.py"]
