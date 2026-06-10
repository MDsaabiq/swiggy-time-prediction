# set the base image
FROM python:3.12-slim

# install lightgbm dependency
RUN apt-get update && apt-get install -y libgomp1

# set up the working directory
WORKDIR /app

# copy the requirements file
COPY requirements_docker.txt ./

# install the packages
RUN pip install -r requirements_docker.txt

# copy the app contents
COPY app.py ./
COPY ./models/preprocessor.joblib ./models/preprocessor.joblib
COPY ./scripts/data_clean_utils.py ./scripts/data_clean_utils.py
COPY ./templates ./templates
COPY ./static ./static
COPY ./run_information.json ./

# expose the port
EXPOSE 8000

# Run the file using command
CMD [ "python","./app.py" ]