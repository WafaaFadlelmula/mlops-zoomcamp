FROM python:3.9.12-slim

RUN pip install -U pip 

RUN pip install pipenv

WORKDIR /app
# Copy Pipfile and Pipfile.lock to container
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY [ "predict.py", "lin_reg.bin", "./"]

ENTRYPOINT [ "waitress-serve", "--host", "0.0.0.0", "--port", "9696", "predict:app" ]

