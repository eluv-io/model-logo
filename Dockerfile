FROM continuumio/miniconda3:latest
WORKDIR /elv

RUN apt-get update && apt-get install -y build-essential \
    && apt-get install -y ffmpeg

RUN \
   conda create -n logo python=3.8 -y

SHELL ["conda", "run", "-n", "logo", "/bin/bash", "-c"]

# Create the SSH directory and set correct permissions
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Add GitHub to known_hosts to bypass host verification
RUN ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts

RUN mkdir logo
COPY setup.py .

ARG SSH_AUTH_SOCK
ENV SSH_AUTH_SOCK ${SSH_AUTH_SOCK}

RUN /opt/conda/envs/logo/bin/pip install .

COPY logo ./logo
COPY config.yml run.py config.py .

COPY weights ./weights

ENTRYPOINT ["/opt/conda/envs/logo/bin/python", "run.py"]FROM continuumio/miniconda3:latest
