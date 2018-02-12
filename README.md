# Face classification and detection 
This repository is largely based on the [B-IT-BOTS robotics team](https://mas-group.inf.h-brs.de/?page_id=622), Real-time face detection and emotion/gender classification using fer2013/IMDB datasets with a keras CNN model and openCV. See: https://github.com/oarriaga/face_classification

They report:
* IMDB gender classification test accuracy: 96%.
* fer2013 emotion classification test accuracy: 66%.

For more information please consult the [technical report](https://github.com/oarriaga/face_classification/blob/master/doc/technical_report.pdf)

Real-time demo:
<div align='center'>
  <img src='images/color_demo.gif' width='400px'>
</div>

## Instructions

### To install


pip install -r requirements.txt

### To train previous/new models for emotion classification:


* Download the fer2013.tar.gz file from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

* Move the downloaded file to the datasets directory inside this repository.

* Untar the file:
> tar -xzf fer2013.tar

* Run the train_emotion_classification.py file
> python3 train_emotion_classifier.py

### To train previous/new models for gender classification:

* Download the imdb_crop.tar file from [here](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) (It's the 7GB button with the tittle Download faces only).

* Move the downloaded file to the datasets directory inside this repository.

* Untar the file:
> tar -xfv imdb_crop.tar

* Run the train_emotion_classification.py file
> python3 train_emotion_classifier.py

### Run real-time emotion demo:
> ./run_live_emo_demo.sh

or

> python3 ./src/video_emotion_color_demo.py

