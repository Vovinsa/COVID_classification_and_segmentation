FROM tensorflow/tensorflow:latest-gpu

ENV PYTHONUNBUFFERED=1
ENV PROJECTDIR /usr/src/app

WORKDIR ${PROJECTDIR}

COPY . .

RUN apt-get install libgl1 -y
RUN pip install pip
RUN pip install -r requirements.txt