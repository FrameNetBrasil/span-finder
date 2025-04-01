FROM python:3.8

WORKDIR /opt/app

# Manually install PyTorch
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install transformers

# Install other dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download (and cache) the encoder
ARG PRETRAINED_ENCODER=xlm-roberta-large
RUN python -c "from transformers import AutoModel; model = AutoModel.from_pretrained('$PRETRAINED_ENCODER');"

# Install jq
RUN apt-get update && apt-get install -y jq

# Setup 'sfpt'
COPY config config/
COPY sftp sftp/
COPY setup.py .
COPY .env.default .env
RUN python setup.py install

# Copy files for DEMO
COPY tools/demo tools/demo/

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
