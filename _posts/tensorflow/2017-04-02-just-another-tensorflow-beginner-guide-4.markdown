---
layout: post
title:  "Just another Tensorflow beginner guide (Part4 - Google Cloud)"
date:   2017-04-02
comments: true
---

Now, let's try train our simple sentiment machine learning model on Google cloud.

[You can checkout the reference code I used in this example from here (click)](https://github.com/liufuyang/kaggle-youtube-8m/tree/master/tf-learn/example-5-google-cloud)

Note that we will use the Tensorflow sample code created in the previous post - [Part 2 - example-3.py]({{ site.baseurl }}{% post_url tensorflow/2017-03-17-just-another-tensorflow-beginner-guide-2 %})

## Prepare your Google Cloud Machine Learning Engine
Firstly follow this guide below
[https://cloud.google.com/ml-engine/docs/quickstarts/command-line](https://cloud.google.com/ml-engine/docs/quickstarts/command-line)

Basically you need to 
* Go to the cloud console [project page](https://console.cloud.google.com/iam-admin/projects)
* Create a project for this tutorial, let's call it `tf-learn-simple-sentiment`. And you do it by click `Create Project` button:

    ![gc-prepare-1](/assets/tensorflow/2017-04-02-just-another-tensorflow-beginner-guide-4/gc-prepare-1.png)*Click the CREATE PROJECT button*

* Enable billing for it if you haven't
* Enable API for this project
    ![gc-prepare-2](/assets/tensorflow/2017-04-02-just-another-tensorflow-beginner-guide-4/gc-prepare-2.png)*Click the continue button*
    
* Install the Gooogle Cloud SDK and login. Just follow the step [here](https://cloud.google.com/ml-engine/docs/quickstarts/command-line) (And you probably have to install Python 2.7 and make an virtual env for it before going further). Pay attention to the service region (I use europe-west1) as later you will specify it when creating bucket or creating training jobs
* Verify process by running `gcloud ml-engine models list` and you should see:
`Listed 0 items.d `


```
python -m tensorflow.tensorboard --logdir=./tmp/example-3 --port=8000 --reload_interval=5
```

### Upload training data into Google Cloud storage

Setup bucket and copy the model input data into it:
```
gsutil mb -l europe-west1 gs://tf-learn-simple-sentiment
gsutil cp -r sentiment_set.pickle gs://tf-learn-simple-sentiment/sentiment_set.pickle
```
After this you should be able to see your data file created at google cloud console.
(Click the `Storage` button from the left panel and make sure the project is selected correctly)

![gc-prepare-3](/assets/tensorflow/2017-04-02-just-another-tensorflow-beginner-guide-4/gc-prepare-3.png)*Bucket created*

https://cloud.google.com/ml-engine/docs/how-tos/getting-started-training-prediction

### Update our model to have training data location as a parameters

As we will later running our model on the cloud, we need to let the model to run with a parameter of the training data location. So that we can test 
the model locally with `--train-files=sentiment_set.pickle`, then on the Google cloud we run the model with `--train-files=gs://tf-learn-simple-sentiment/sentiment_set.pickle`.

In order to do this, we update our `example5.py` code by creating 
a function called `train_modle()` which contains most part of the 
code and add a main function to call on this `train_modle` function:

```python
# example5.py
import ...
from tensorflow.python.lib.io import file_io
...

def train_model(train_file='sentiment_set.pickle', **args):
    # Here put all the main training code in this function
    file_stream = file_io.FileIO(train_file, mode='r')
    train_x, train_y, test_x, test_y = pickle.load(file_stream)
    ...
    ...

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
      '--train-file',
      help='GCS or local paths to training data',
      required=True
    )

    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__
    job_dir = arguments.pop('job_dir')
    
    train_model(**arguments)
```
Note that in order to let the code can access `gs://file` file format from Google Cloud Storage,
we need to use a different library to open the file `tensorflow.python.lib.io.file_io.FileIO(train_file, mode='r')` (as suggested on a stack-overflow [answer](http://stackoverflow.com/questions/42761075/reading-files-in-google-cloud-machine-learning) or [answer](http://stackoverflow.com/questions/41633748/load-numpy-array-in-google-cloud-ml-job))

Also note that we need to add `--job-dir` as a parameter here, not because 
our code needs but, it is because later when running on cloud, google 
cloud will call on our `example5.py` with this parameter. To make it work we have to just take in this parameter here for now.

### Running locally

* Firstly verify now that you can run the python training code manually via python command:
    ```
    # under folder example-5-google-cloud, run
    $ python -m trainer.example5 --train-file=sentiment_set.pickle --job-dir=./tmp/example-5
    ```

* If that works, now try running it again locally via Google Cloud SDK tool `gcloud` just to verify the code again:
    ```
    # under folder example-5-google-cloud, run
    $ gcloud ml-engine local train \
      --module-name trainer.example5 \
      --package-path ./trainer \
      -- \
      --train-file sentiment_set.pickle \
      --job-dir ./tmp/example-5
    ```
    (Alternatively, run the `gcloud.local.run.sh` script, it has the same contents.)

If it all works, you should see some model training output just as if 
you normally run the example5.py model locally using python command above.

And you should be able to use the tensorboard to see the output as well:
```
python -m tensorflow.tensorboard --logdir=./tmp/example-5/logs --port=8000 --reload_interval=5
```

Then you should be all set for running the job on the Google Cloud.

### Running on cloud
* Firstly create a config file `config=trainer/cloudml-gpu.yaml` under folder trainer:
    ```
    trainingInput:
    scaleTier: CUSTOM
    # standard_gpu provides 1 GPU. Change to complex_model_m_gpu for 4 GPUs
    masterType: standard_gpu
    runtimeVersion: "1.0"
    ```

* Then running the code on the cloud should be simple now as we have prepared everything for it,
simple run command like this:
(Note that you probably will have to change the `REGION=europe-west1` to be the same region you chose when you creating the bucket in previous steps)
    ```
    export BUCKET_NAME=tf-learn-simple-sentiment
    export JOB_NAME="example_5_train_$(date +%Y%m%d_%H%M%S)"
    export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
    export REGION=europe-west1

    gcloud ml-engine jobs submit training $JOB_NAME \
      --job-dir gs://$BUCKET_NAME/$JOB_NAME \
      --runtime-version 1.0 \
      --module-name trainer.example5 \
      --package-path ./trainer \
      --region $REGION \
      --config=trainer/cloudml-gpu.yaml \
      -- \
      --train-file gs://tf-learn-simple-sentiment/sentiment_set.pickle
    ```
    (Alternatively, run `source gcloud.remote.run.sh` script, it has the same contents.)

Then you should be seeing some logs output in your local terminal. The same log can be found on the google cloud console as well.
Also the console log is showing a Teska K80 prepared for training the model:

![gc-run-1](/assets/tensorflow/2017-04-02-just-another-tensorflow-beginner-guide-4/gc-run-1.png)
(I am not sure why those logs are marked as ERROR. Please comment if you've got some clue on it.)

![gc-run-1-1](/assets/tensorflow/2017-04-02-just-another-tensorflow-beginner-guide-4/gc-run-1-1.png)*Logging view on google cloud*

On the storage tab of the cloud console, noticing that those model summary logs are also written out in
specified folders:

![gc-run-2](/assets/tensorflow/2017-04-02-just-another-tensorflow-beginner-guide-4/gc-run-2.png)*Storage view on google cloud*

* And then you can use the tensorboard to view the training logs and graph
by doing:
```
$ echo tensorboard log dir: ${JOB_DIR}
$ tensorboard --logdir=${JOB_DIR}/logs --port 8000 --reload_interval=5
```
(Alternatively running script `gcloud.tensorboard.show.sh` from example code base)

Open browser on localhost:8000, 
you should be able to see the loss and accuracy info as I see them now:

![gc-run-3](/assets/tensorflow/2017-04-02-just-another-tensorflow-beginner-guide-4/gc-run-3.png)

If you could not see any results, make sure in your code that the
tf.summary.FileWriter's flush function is called in the end to 
for sure output those logs. 
```
# call this after writer add all the summaries
writer.flush()
```
([Info here](https://github.com/tensorflow/tensorflow/issues/2353))

### Summary
Contragulations, now you have made a simple Tensorflow model and trained on Google Cloud with a Tesla K80 graphic card. 

From here you should be capable of using a much larger dataset and perhaps a much more complex model for some doing real fancy applications :)
