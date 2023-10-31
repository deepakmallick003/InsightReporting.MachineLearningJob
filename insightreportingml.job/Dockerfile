# 
FROM python:3.9


# 
WORKDIR /app

# 
COPY insightreportingml.job/requirements.txt ./


ENV DEBIAN_FRONTEND=noninteractive
ENV ACCEPT_EULA=Y

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    unixodbc-dev \
    curl \
    gnupg \
    apt-transport-https \
    apt-utils \
    && curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/9/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends msodbcsql17 mssql-tools --fix-missing \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip \
    && pip3 install pyodbc

# 
RUN pip install --no-cache-dir --upgrade -r requirements.txt

#
RUN python -m nltk.downloader stopwords

#
RUN python -m nltk.downloader wordnet

#
RUN python -m nltk.downloader punkt

#
RUN python -m nltk.downloader averaged_perceptron_tagger

# 
COPY insightreportingml.job/. ./

# 
CMD [ "python", "main.py" ]
