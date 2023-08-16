# 
FROM python:3.9

# 
WORKDIR /app

# 
COPY insightreportingml.job/requirements.txt ./

# 
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 
COPY insightreportingml.job/. ./

# 
CMD [ "python", "main.py" ]
