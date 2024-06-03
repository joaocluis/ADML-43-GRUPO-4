FROM python:3.9.19-alpine3.20

WORKDIR /app

RUN apk update && apk add --no-cache \
    build-base \
    gfortran \
    py3-pip \
    openblas-dev

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "./condensado.py"]
