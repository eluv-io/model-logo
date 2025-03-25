FROM continuumio/miniconda3:latest
WORKDIR /elv

RUN conda create -n mlpod python=3.8 -y

SHELL ["conda", "run", "-n", "mlpod", "/bin/bash", "-c"]

RUN apt-get update && apt-get install -y build-essential && apt-get install -y ffmpeg

# Create the SSH directory and set correct permissions
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Add GitHub to known_hosts to bypass host verification
RUN ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts

ARG SSH_AUTH_SOCK
ENV SSH_AUTH_SOCK ${SSH_AUTH_SOCK}

RUN mkdir logo
COPY setup.py .

RUN /opt/conda/envs/mlpod/bin/pip install .

COPY weights ./weights

COPY logo ./logo
COPY config.yml run.py config.py .

ENTRYPOINT ["/opt/conda/envs/mlpod/bin/python", "run.py"]
