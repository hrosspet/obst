FROM tensorflow/tensorflow:latest-py3-jupyter

COPY . /obst

RUN pip install -r /obst/requirements.txt
RUN pip install -e /obst/

CMD [ "python3", "/obst/obst/__main__.py" ]