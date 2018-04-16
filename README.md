# FaceCheck
# An Emotion Detector by Facial Expression for User Experence Evaluation
This project won the 2nd runner-up in 2017 Hang-Seng Bank Fintech

1) Face Detector:

Use 'detector.py' and xml file to detect human face in arbitary photos.

2) Facial Expression Recognition: 

Use pretrained googlenet or vggnet to finetune on facial expression data.

training dataset: facial expression challenge on Kaggle

AgDgSd_Hap_Ner.py: Merge facial expression of 'Angry', 'Disgust' and 'Sad'

AgDg_Hap_Sad_Ner.py: Merge facial expession of 'Angry', 'Disgust' as one category

Happy_Angry_Neural.py: Finetune vggnet using 'Happy', 'Angry' and 'Neutral' only

detector.py: Detect human face in arbitrary photos 

face_googlenet.py: Finetune on googlenet

vgg16_face.py: Finetune on vgg16net
