---
layout: post
title:  "Just another Tensorflow beginner guide (Part5 - Deploy a Keras Model)"
date:   2017-04-07
comments: true
---

Just to make this tutorial series a bit more useful, let's try if we could deploy our
previously made Keras model onto Google Cloud.

To do this, I found some good info from [this post](https://medium.com/google-cloud/keras-inception-v3-on-google-compute-engine-a54918b0058)

It seems like we need these a few steps to make things work:
* Make sure the model file and lexicon file are located on google cloud
* Write a python local service and make it work
* Dockerize it and run the docker on google cloud

## Prepare needed files

* `model.h5` file should be already uploaded to Google Cloud Storage if you have followed the 
previous steps. Downlaod it to your local folder.
* `lexicon.pickles` is a file that will be used to process our input data. It is 
generated while running script `create_sentiment_featuresets.py`. Check [post here](http://liufuyang.github.io/2017/03/17/just-another-tensorflow-beginner-guide-2.html) and [code here](https://github.com/liufuyang/kaggle-youtube-8m/tree/master/tf-learn/example-3-sentiment).


Then we can wrote some simple python code to see if our model works:

Firstly create a predictor.py
```python
# predictor.py

import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class Predictor:
    def __init__(self, model, lexicon):
        self.model = model
        self.lexicon = lexicon
        self.lemmatizer = WordNetLemmatizer()
    
    def _get_feature_vector(self, input_str_list):
        featureset = np.empty((0, len(self.lexicon)))
        
        for i, input_str in enumerate(input_str_list):
            current_words = word_tokenize(input_str.lower())
            current_words = [self.lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(self.lexicon))

            for word in current_words:
                if word.lower() in self.lexicon:
                    index_value = self.lexicon.index(word.lower())
                    features[index_value] += 1
            features = features / np.max(features)
            featureset = np.append(featureset, np.array([features]), axis=0)
            
        return featureset
    
    def predict(self, input_str_list):
        
        featureset = self._get_feature_vector(input_str_list)
        
        assert featureset.shape[0] > 0
        
        result = self.model.predict(featureset)
        
        return result
```

Then try some simple test with it main0.py
```python
# main0.py
import pickle

from predictor import Predictor
from keras.models import load_model

lexicon_file = open('lexicon.pickle', mode='r')
lexicon = pickle.load(lexicon_file)
print(len(lexicon))


model = load_model('model.h5')
print(model.summary())

predictor = Predictor(model, lexicon)

result = predictor.predict(['hehe this is really good, I love it', "well, I don't know"])

print(result)

result = predictor.predict(['hehe this is really bed, I hate it'])

print(result)
```

The above code will load the model and lexicon and do some predictions on some 
example sentences. Note that those sentences are fed in as a list. Thus we are 
expecting the return will also be as in list.

Run the code above and the output should look like this:
```
$ python main0.py
Using TensorFlow backend.
423
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 32)                13568
_________________________________________________________________
dropout_1 (Dropout)          (None, 32)                0
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 66
=================================================================
Total params: 13,634
Trainable params: 13,634
Non-trainable params: 0
_________________________________________________________________
None
[[ 0.95534408  0.04465592]
 [ 0.27301568  0.72698432]]
[[ 0.08944085  0.91055912]]
```

The result seems pretty okay, if you compare it with the sentences fed in.

## Make it as a service

Now we have some example code working, let's turn it into a web service that 
can take http requests and give result via http response. We can do this via
making a main.py as below:

```python
# main.py
from flask import Flask, current_app, request, jsonify
import io
import pickle
from predictor import Predictor
import logging

from keras.models import load_model

lexicon_file = open('lexicon.pickle', mode='r')
lexicon = pickle.load(lexicon_file)

model = load_model('model.h5')
model.summary()
predictor = Predictor(model=model, lexicon=lexicon)

app = Flask(__name__)

@app.before_first_request
def setup_logging():
    if not app.debug:
        # In production mode, add log handler to sys.stderr.
        app.logger.addHandler(logging.StreamHandler())
        app.logger.setLevel(logging.INFO)

@app.route('/', methods=['POST'])
def predict():
    data = {}
    try:
        input_str_list = request.get_json()['sentences']
    except Exception:
        return jsonify(status_code='400', msg='Bad Request'), 400

    predictions = predictor.predict(input_str_list).tolist()
    current_app.logger.info('Predictions: %s', predictions)
    return jsonify(predictions=predictions)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
```

Keep the code running and do an API call via another terminal, you should see 
the returned result:
```
$ curl -H "Content-Type: application/json" -X POST -d '{"sentences": ["I love this movie and enjoy watching it"] }' http://localhost:8080

{
  "predictions": [
    [
      0.8991053700447083,
      0.10089470446109772
    ]
  ]
}
```

## Put everything in docker
We start this by creating some server config files and a dockerfile:

`nginx.conf`
```
daemon off;
error_log /dev/stdout info;
worker_processes 1;

# user nobody nogroup;
pid /tmp/nginx.pid;

events {
    worker_connections 1024;
    accept_mutex off;
}

http {
    include mime.types;
    default_type application/octet-stream;
    access_log /dev/stdout combined;
    sendfile on;

    upstream app_server {
        # For a TCP configuration:
        server 127.0.0.1:5000 fail_timeout=0;
    }

    server {
        listen 80 default;
        client_max_body_size 4G;
        server_name _;

        keepalive_timeout 5;

        # path for static files
        root /opt/app/static;

        location / {
            # checks for static file, if not found proxy to app
            try_files $uri @proxy_to_app;
        }

        location @proxy_to_app {
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header Host $http_host;
            proxy_redirect off;

            proxy_pass   http://app_server;
        }

    }
}
```

`supervisord.conf`
```
[supervisord]
nodaemon = true

[program:nginx]
command = /usr/sbin/nginx
startsecs = 60
stdout_events_enabled = true
stderr_events_enabled = true

[program:app-gunicorn]
command = /opt/venv/bin/gunicorn main:app -w 3 -b 0.0.0.0:5000 --log-level=info --chdir=/opt/app -t 150 
;command = /opt/venv/bin/python /opt/app/main.py
autostart= true
autorestart= true
stdout_events_enabled = true
stderr_events_enabled = true

[eventlistener:stdout]
command = supervisor_stdout
buffer_size = 1000
events = PROCESS_LOG
result_handler = supervisor_stdout:event_handler
```

`Dockerfile`
```Dockerfile
FROM ubuntu:16.04

RUN apt-get update --fix-missing

# Install virtualenv, nginx, supervisor
RUN apt-get install -y python-pip python-virtualenv
RUN apt-get install -y nginx supervisor

RUN service supervisor stop

# create virtual env and install dependencies
RUN virtualenv /opt/venv

RUN /opt/venv/bin/pip install tensorflow==1.0.1
RUN /opt/venv/bin/pip install keras==2.0.3
RUN /opt/venv/bin/pip install h5py
RUN /opt/venv/bin/pip install nltk
RUN /opt/venv/bin/python -m nltk.downloader punkt wordnet
RUN /opt/venv/bin/pip install flask gunicorn

# expose port
EXPOSE 80

RUN pip install supervisor-stdout

# Add our config files
ADD ./supervisord.conf /etc/supervisord.conf
ADD ./nginx.conf /etc/nginx/nginx.conf

# Copy our service code
ADD ./service /opt/app

# restart nginx to load the config
RUN service nginx stop

# start supervisor to run our wsgi server
CMD supervisord -c /etc/supervisord.conf -n
```

Build and test locally:
```
$ docker build -t sentiment-service .
$ docker run -it --rm -p 8081:80 sentiment-service
```
(Do the same curl command above to test but note to change the port to 8081)

Then use the following script to push the image onto google cloud platform:
```
#/bin/sh
docker build -t sentiment-service .
docker tag sentiment-service gcr.io/PROJECT_ID/sentiment-service
gcloud docker -- push gcr.io/PROJECT_ID/sentiment-service
```
Note that you have to put your own `PROJECT_ID` there, for me as my google cloud
project is called `tf-learn-simple-sentiment` so I use that instead.

You’ve built and pushed the docker image to Google Container Registry. From here, create the server and pull down the previously created docker image. First, you’ll enable the API

[https://console.developers.google.com/apis/api/compute_component/overview?project=PROJECT_ID](https://console.developers.google.com/apis/api/compute_component/overview)

```
$ gcloud compute firewall-rules create default-allow-http --allow=tcp:80 --target-tags http-server
$ gcloud compute instances create sentiment-service --machine-type=n1-standard-1 --zone=europe-west1-b --tags=http-server
```

This will create an instance with the following info:
```
NAME               ZONE            MACHINE_TYPE   PREEMPTIBLE  INTERNAL_IP  EXTERNAL_IP   STATUS
sentiment-service  europe-west1-b  n1-standard-1               10.132.0.2   35.187.64.91  RUNNING
```

Then ssh into the instance and install docker, pull image:
```
$ gcloud compute ssh sentiment-service --zone=europe-west1-b
$ curl -sSL https://get.docker.com | sh
$ sudo gcloud docker pull -- gcr.io/tf-learn-simple-sentiment/sentiment-service:latest
```

Finally run the service 
```
$ sudo docker run -td -p 80:80 gcr.io/tf-learn-simple-sentiment/sentiment-service
```

Then you can test your service as:
```
$ curl -H "Content-Type: application/json" -X POST -d '{"sentences": ["I love this movie and enjoy watching it"] }' http://35.187.64.91

{
  "predictions": [
    [
      0.8991053700447083,
      0.10089470446109772
    ]
  ]
}
```

Congratulations, now you have a sentiment analysis machine learning Keras model running 
as an API service on google cloud.
