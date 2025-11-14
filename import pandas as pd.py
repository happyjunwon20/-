import pandas as pd
import numpy as np
import os
from gtts.lang import tts_langs
from gtts import gTTS
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator
'''pd.DataFrame({"이름" : ["철수", "영희"], "성별" : ["여", "남"]}, index = ["name", "gender"])
heart = np.array([[0, 1, 0, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]])
plt.figure(figsize = (4, 4))
plt.imshow(heart, cmap = 'gray')
plt.axis("off")
plt.tight_layout()
plt.show()
temp = np.array([28.5, 29.0, 30.2, 33.1, 35.0, 36.5, 34.2])
hot = temp[temp >= 33]
hot_num = len(hot)
print(hot.tolist())
print(GoogleTranslator(source = "auto", target = "ko").translate("Bonjour tout le monde"))
tts = gTTS(text = "안녕하세요", lang = 'ko')
tts.save("hello.mp3")
os.system("start hello.mp3")
bun = input("입력: ")
bun_tr = GoogleTranslator(source = "ko", target = "en").translate(bun)
print(bun_tr)
tts2 = gTTS(text = bun_tr, lang = "en")
tts2.save("bunuk.mp3")
os.system("start bunuk.mp3")'''
from sklearn.linear_model import LinearRegression
'''x = [[2], [4], [6], [8], [10]]
y = [[81], [83], [90], [97], [100]]
plt.scatter(x, y)
plt.show()
model = LinearRegression()
model.fit(x, y)
result = model.predict([[7]])
print(result)'''
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

'''iris = load_iris()
X = iris.data
Y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))'''

'''seoul = pd.read_csv("weather.csv", encoding="cp949")
print(seoul.head())
print(seoul.describe())

seoul.drop('지점',axis=1,inplace=True)
seoul.drop('지점명',axis=1,inplace=True)
print(seoul.head())

seoul.columns=['날짜','평균기온','최저기온','최고기온']

seoul['날짜'] = pd.to_datetime(seoul['날짜'])

print(seoul.isnull().sum())

seoul.dropna(subset=['최저기온'],axis=0,inplace=True)
seoul.dropna(subset=['최고기온'],axis=0,inplace=True)

seoul['년도']=seoul['날짜'].dt.year
print(seoul.info())


conditions=(seoul['날짜'].dt.month==8) & (seoul['날짜'].dt.day==15)
print(conditions)
seoul0815=seoul[conditions]
print(seoul0815)

fig = plt.figure(figsize=(15,7))
plt.rc('font', family='Malgun Gothic')
X = seoul0815[['년도']]
Y = seoul0815['평균기온']
plt.xlabel('년도')
plt.ylabel('평균기온')
plt.scatter(X,Y)
plt.show()

model = LinearRegression()
model.fit(X, Y)
print(model.predict([[2025]]))

X = seoul0815["년도"]
Y = seoul0815["평균기온"]

fp1 = np.polyfit(X, Y, 2)
f1 = np.poly1d(fp1)
fx = np.linspace(2015, 2024)
plt.figure(figsize = (15, 7))
plt.scatter(X, Y)
plt.plot(fx, f1(fx), ls = "dashed", lw = 3, color = "g")

plt.xlabel("년도")
plt.ylabel("평균기온")
plt.show()

X = seoul0815[["년도", "최저기온", "최고기온"]]
Y = seoul0815["평균기온"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = 0.7, test_size = 0.3)

model = LinearRegression()
model.fit(x_train, y_train)
y_pr = model.predict(x_test)

plt.plot(model.predict(x_test[:50]), label = "predict")
plt.plot(y_test[:50].values.reshape(-1, 1), label = "real temp")
plt.legend()
plt.figure(figsize = (15, 7))

plt.scatter(y_test, y_pr, alpha = 0.4)
plt.show()
print(model.score(x_train, y_train))'''

'''import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

import tensorflow as tf
import numpy as np
from PIL import Image

def predict_digit(model, image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))

    img_array = np.array(img)
    img_array = img_array / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_label = np.argmax(predictions)
    
    print("예측된 숫자:", predicted_label)
    return predicted_label
predict_digit(model, "7.png")'''
'''import os, zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models'''
'''

local_zip = r"C:\data\cats_and_dogs_filtered.zip"
extract_to = r"C:\data"



# 압축 해제(이미 해제했다면 생략)
with zipfile.ZipFile(local_zip, 'r') as zf:
    zf.extractall(extract_to)

base_dir = os.path.join(extract_to, 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_gen = ImageDataGenerator(rescale=1/255)
val_gen = ImageDataGenerator(rescale=1/255)

train_generator = train_gen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

val_generator = val_gen.flow_from_directory(
    validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=5,
    validation_data=val_generator,
    validation_steps=50
)

model.save("animals.h5")
print("animals.h5 로 저장되었습니다.")

import os
import numpy as np
import tensorflow as tf'''
'''from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model

model_path = "animals.h5"
if not os.path.exists(model_path):
    print("모델 파일이 없습니다.")
    exit()

model = load_model(model_path)
print("모델 로드 완료")

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"[오류] 파일이 없습니다: {image_path}")
        return

    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        print(f"{image_path} → 강아지일 확률: {prediction:.2f}")
    else:
        print(f"{image_path} → 고양이일 확률: {1 - prediction:.2f}")


predict_image("test1.jpg" )'''

'''import cv2
 
img = cv2.imread('image1.png', cv2.IMREAD_COLOR)
 
cartoon_img = cv2.stylization(img, sigma_s=100, sigma_r=0.9)  
 
cv2.imshow('original', img)
cv2.imshow('cartoon', cartoon_img)  
cv2.waitKey(0)  
cv2.destroyAllWindows() 
 
cv2.imwrite('img2_cartoon.jpg', cartoon_img)'''

'''import cv2'''

'''img = cv2.imread('image1.png', cv2.IMREAD_COLOR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray", gray)'''
'''
import cv2

img = cv2.imread('image1.png', cv2.IMREAD_COLOR)

gray_sketch, color_sketch = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)

cv2.imshow('Original', img)
cv2.imshow('Pencil Sketch (Gray)', gray_sketch)
cv2.imshow('Pencil Sketch (Color)', color_sketch)

cv2.imwrite('img2_pencil_gray.jpg', gray_sketch)
cv2.imwrite('img2_pencil_color.jpg', color_sketch)

cv2.waitKey(0)
cv2.destroyAllWindows()'''
'''import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

while(True):
    ret, frame = cap.read() 
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.05,5) 
    print("Number of faces detected: " + str(len(faces)))

    if len(faces):
        for (x,y,w,h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, dsize=(0, 0), fx=0.04, fy=0.04)
            face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA)
            frame[y:y+h, x:x+w] = face_img

    cv2.imshow('result', frame)
        
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()'''
'''import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=5, validation_split=0.1)


loss, acc = model.evaluate(x_test, y_test)
print(f"테스트 정확도: {acc:.4f}")


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(8, 4))
for i in range(5):
    image = x_test[i]
    label = y_test[i]
    pred = model.predict(image.reshape(1, 28, 28, 1)).argmax()

    plt.subplot(1, 5, i+1)
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f"예측: {class_names[pred]}")
    plt.axis('off')
plt.tight_layout()
plt.show()'''

'''from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()'''
'''
import pygame
import random
import sys

pygame.init()

WIDTH, HEIGHT = 600, 400
CELL_SIZE = 20
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("스네이크 게임 - 방향키 조작")

BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

snake = [(100, 100)]
directions = {
    "UP": (0, -CELL_SIZE),
    "DOWN": (0, CELL_SIZE),
    "LEFT": (-CELL_SIZE, 0),
    "RIGHT": (CELL_SIZE, 0)
}
direction = "RIGHT"
opposite = {"UP":"DOWN", "DOWN":"UP", "LEFT":"RIGHT", "RIGHT":"LEFT"}
apple = (random.randint(0, WIDTH // CELL_SIZE - 1) * CELL_SIZE,
         random.randint(0, HEIGHT // CELL_SIZE - 1) * CELL_SIZE)

clock = pygame.time.Clock()

def draw_snake():
    for segment in snake:
        pygame.draw.rect(screen, GREEN, (*segment, CELL_SIZE, CELL_SIZE))

def draw_apple():
    pygame.draw.rect(screen, RED, (*apple, CELL_SIZE, CELL_SIZE))

def move_snake():
    global apple
    head_x, head_y = snake[0]
    dx, dy =  directions[direction]
    new_head = (head_x + dx, head_y + dy)

    if (new_head in snake or
        new_head[0] < 0 or new_head[0] >= WIDTH or
        new_head[1] < 0 or new_head[1] >= HEIGHT):
        pygame.quit()
        sys.exit()

    snake.insert(0, new_head)

    if new_head == apple:
        apple = (random.randint(0, WIDTH // CELL_SIZE - 1) * CELL_SIZE,
                 random.randint(0, HEIGHT // CELL_SIZE - 1) * CELL_SIZE)
    else:
        snake.pop()
def ai_di():
    global direction
    head_x, head_y = snake[0]
    apple_x, apple_y = apple
    if head_x < apple_x and direction != "LEFT":
        direction = "RIGHT"
    elif head_x > apple_x and direction != "RIGHT":
        direction = "LEFT"
    elif head_y > apple_y and direction != "DOWN":
        direction = "UP"
    elif head_y < apple_y and direction != "UP":
        direction = "DOWN"
while True:
    screen.fill(BLACK)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:    cand = "UP"
            elif event.key == pygame.K_DOWN: cand = "DOWN"
            elif event.key == pygame.K_LEFT: cand = "LEFT"
            elif event.key == pygame.K_RIGHT: cand = "RIGHT"
            else:
                cand = None

            if cand is not None:
                if len(snake) == 1 or cand != opposite[direction]:
                    direction = cand

    move_snake()
    draw_snake()
    draw_apple()
    ai_di()

    pygame.display.flip()
    clock.tick(10)'''
'''  
from urllib.request import urlopen
from urllib.error import HTTPError
from urllib.parse import urlparse, urljoin, unquote
from requests import get, post
from os import makedirs, remove
from os.path import join, exists
from shutil import rmtree
from json import load, loads
from zipfile import ZipFile
from pandas import DataFrame
from ydf import load_model



class MLforKidsNumbers:
    def __init__(self, key=None, modelurl=None):
        self._scratchkey = key

        if modelurl is not None:
            self._message("Checking for downloaded model...")
            key = self._get_model_key(modelurl)
            model_folder = self._get_saved_model_folder(key)
            if exists(model_folder):
                self._message("Reusing downloaded model from " + model_folder)
            else:
                self._download_model(modelurl, model_folder)

            self._message("Loading model...")
            self.MODEL = load_model(model_folder)

            self._message("Accessing model metadata...")
            self.METADATA = self._read_json_file(join(model_folder, "mlforkids.json"))
            self._message("Model trained at " + self.METADATA["lastupdate"])
        else:
            self.MODEL = None


    def has_model(self):
        return self.MODEL is not None


    # ------------------------------------------------------------
    #  Helper functions to display output
    # ------------------------------------------------------------

    def _message(self, str):
        print("\033[1m MLforKids : " + str + " \033[0m")

    def _debug(self, str):
        print("-----------------------------------------------------")
        print(str)
        print("-----------------------------------------------------")


    # ------------------------------------------------------------
    #  Get the key from a model URL
    # ------------------------------------------------------------

    def _get_model_key(self, url):
        parsed_url = urlparse(url)
        path_segments = parsed_url.path.split('/')
        if len(path_segments) > 2:
            return unquote(path_segments[2])
        raise Exception("Unrecognised URL")


    # ------------------------------------------------------------
    #  Identify the location where project files will be saved
    # ------------------------------------------------------------

    def _get_saved_model_folder(self, key):
        folder = join("saved_models", key)
        return folder


    # ------------------------------------------------------------
    #  Helper function to read a JSON file
    # ------------------------------------------------------------

    def _read_json_file(self, location):
        with open(location, 'r') as file:
            return loads(file.read())


    # ------------------------------------------------------------
    #  Helper function to download a file to disk
    # ------------------------------------------------------------

    def _download_file(self, url, target):
        headers = {'User-Agent': 'MachineLearningForKids-Python'}
        response = get(url, headers=headers)
        if response.status_code == 200:
            with open(target, 'wb') as file:
                file.write(response.content)
        else:
            raise Exception("Failed to download file from {url}")


    # ------------------------------------------------------------
    #  Get URL location of zip file on model server
    # ------------------------------------------------------------

    def _get_model_info(self, status_url):
        try:
            with urlopen(status_url) as url:
                project_info = load(url)

                if project_info["status"] != "Available":
                    self._debug(project_info)
                    raise Exception ("The model is not available for use - the current status is " + project_info["status"])

                return project_info

        except HTTPError as e:
            if e.code == 404:
                self._message("The model is no longer available on the model server.")
                self._message("Models are only stored online for a short time. ")
                self._message("Train a new model on the Machine Learning for Kids site, then try again with the new URL.")
                raise Exception ("Model unavailable")
            else:
                raise e


    # ------------------------------------------------------------
    #  Download the model zip from the model server and unpack
    # ------------------------------------------------------------

    def _download_model(self, status_url, model_folder):
        self._message("Getting model info...")
        model_info = self._get_model_info(status_url)

        self._message("Preparing for download...")
        if exists(model_folder):
            rmtree(model_folder)
        makedirs(model_folder)

        self._message("Downloading model...")
        model_zip = join(model_folder, "model.zip")
        self._download_file(
            urljoin(model_info["urls"]["model"], "model.zip"),
            model_zip)
        self._download_file(
            status_url,
            join(model_folder, "mlforkids.json"))

        self._message("Unpacking model...")
        with ZipFile(model_zip, 'r') as zip_ref:
            zip_ref.extractall(model_folder)
        remove(model_zip)



    def _sort_by_confidence (self, e):
        return e["confidence"]



    #
    # This function will store your data in one of the training
    # buckets in your machine learning project
    #
    #  key - API key - the secret code for your ML project
    #  data - the data that you want to store as a training example
    #  label - the training bucket to put the example into
    #
    def store(self, data, label):
        if self._scratchkey is None:
            self._message("You need to provide a key to be able to add to your training data")
            self._message("This can only be done for projects that are stored in the cloud")
            raise Exception ("Key unavailable")

        url = ("https://machinelearningforkids.co.uk/api/scratch/" +
               self._scratchkey +
               "/train")

        response = post(url, json={ "data" : data, "label" : label })
        if response.ok == False:
            # if something went wrong, display the error
            print(response.json())




    # use the model to classify the provided data
    #  returns a sorted list of objects, one for each label
    #  each with a confidence percentage
    def classify(self, data):
        if self.MODEL is None:
            self._message("Train a new model on the Machine Learning for Kids site, then try again with the new URL.")
            raise Exception ("Model unavailable")

        labelled = {}
        types = {}
        for feature in self.METADATA["features"]:
            label = self.METADATA["features"][feature]["name"]
            type = self.METADATA["features"][feature]["type"]
            if label == "mlforkids_outcome_label":
                continue
            if not feature in data:
                raise Exception("Missing required value " + feature)
            labelled[label] = data[feature]
            if type != "object":
                types[label] = type
        df = DataFrame([ labelled ])
        classifications = self.MODEL.predict(df)
        results = []
        if len(self.METADATA["labels"]) == 2:
            results.append({
                "class_name" : self.METADATA["labels"][0],
                "confidence" : int((1 - classifications[0].item()) * 100)
            })
            results.append({
                "class_name" : self.METADATA["labels"][1],
                "confidence" : int((classifications[0].item()) * 100)
            })
        else:
            idx = 0
            for classification in classifications[0]:
                results.append({
                    "class_name": self.METADATA["labels"][idx],
                    "confidence": int(classification.item() * 100)
                })
                idx += 1

        results.sort(reverse=True, key=self._sort_by_confidence)
        return results
    
# this module is used to create the game user interface
import pygame
# this module is used to make HTTP requests to your machine learning model
import requests
# this module is used to choose a random colour for the user interface and
#  make random choices about moves the computer should make
import random
# this module is used to interact with your machine learning project
from mlforkidsnumbers import MLforKidsNumbers


project = MLforKidsNumbers(
    # keys and URLs specific to your project will be added here
)




############################################################################
# Constants that match names in your Machine Learning project
############################################################################

# descriptions of the contents of a space on the game board
EMPTY = "EMPTY"
OPPONENT = "OPPONENT"   # for the human, the OPPONENT is the computer
                        # for the computer, the OPPONENT is the human
PLAYER = "PLAYER"       # for the human, the PLAYER is the human
                        # for the computer, the PLAYER is the computer

# descriptions of the locations on the game board
top_left = "top_left"
top_middle = "top_middle"
top_right = "top_right"
middle_left = "middle_left"
middle_middle = "middle_middle"
middle_right = "middle_right"
bottom_left = "bottom_left"
bottom_middle = "bottom_middle"
bottom_right = "bottom_right"

#
############################################################################




############################################################################
# Converting between labels and numeric values
############################################################################

# training examples refer to a location on the game board
deconvert = {}
deconvert[top_left] = 0
deconvert[top_middle] = 1
deconvert[top_right] = 2
deconvert[middle_left] = 3
deconvert[middle_middle] = 4
deconvert[middle_right] = 5
deconvert[bottom_left] = 6
deconvert[bottom_middle] = 7
deconvert[bottom_right] = 8




############################################################################
# Machine Learning functions
############################################################################

# who the two players are
HUMAN = "HUMAN"
COMPUTER = "COMPUTER"

# Storing a record of what has happened so the computer can learn from it!
#   contents of the board at each stage in the game
gamehistory = {
    HUMAN : [],
    COMPUTER : []
}
#   decisions made by each player
decisions = {
    HUMAN : [],
    COMPUTER : []
}



# Use your machine learning model to decide where the
#   computer should move next.
#
#  board :  list of board spaces with the current state of each space
#      e.g.  [ HUMAN, COMPUTER, HUMAN, EMPTY, EMPTY, HUMAN, COMPUTER, HUMAN, COMPUTER ]
def classify(board):
    debug("Predicting the next best move for the computer")

    if project.has_model():
        # get the current state of the game board
        state = get_board_from_perspective(board, COMPUTER)
        testvalue = {
            "TopLeft" : state[0],
            "TopMiddle" : state[1],
            "TopRight" : state[2],
            "MiddleLeft" : state[3],
            "MiddleMiddle" : state[4],
            "MiddleRight" : state[5],
            "BottomLeft" : state[6],
            "BottomMiddle" : state[7],
            "BottomRight" : state[8]
        }
        # send the state of the game board to your machine learning model
        predictions = project.classify(testvalue)

        # responseData will contain the list of predictions made by the
        #  machine learning model, starting from the one with the most
        #  confidence, to the one with the least confidence
        for prediction in predictions:
            # we can't make a move unless the space is empty, so
            #  check that first
            if is_space_empty(board, prediction["class_name"]):
                return prediction

    # If we're here, it means that we don't have a machine learning model,
    #  or possibly none of the predictions made by the model were
    #  actually empty!

    # Pick a random space to move in
    spaces = list(deconvert.keys())
    for space in random.sample(spaces, len(spaces)):
        # we can't make a move unless the space is empty, so
        #  check that first
        if is_space_empty(board, space):
            return { "class_name" : space }



# Add a move that resulted in a win to the training data for the
# machine learning model
#
#  board         :  list of board spaces with the current state of each space
#      e.g.  [ HUMAN, COMPUTER, HUMAN, EMPTY, EMPTY, HUMAN, COMPUTER, HUMAN, COMPUTER ]
#  who           :  whose training data this is
#      e.g.    HUMAN
#  name_of_space :  name of the space that the move was in
#      e.g.    bottom_left
def add_to_train(board, who, name_of_space):
    print ("Adding the move in %s by %s to the training data" % (name_of_space, who))

    # convert the contents of the board into a list of whose symbol
    #   is in that space, from the perspective of 'who'
    #  e.g. [ PLAYER, OPPONENT, PLAYER, EMPTY, EMPTY, PLAYER, OPPONENT, PLAYER, OPPONENT ]
    data = get_board_from_perspective(board, who)

    # the location that they chose to make a move in
    label = name_of_space

    # add move to the machine learning project training data
    project.store(data, label)





# Someone won the game.
#  A machine learning model could learn from this...
#
#  winner          : who won - either HUMAN or COMPUTER
#  boardhistory    : the contents of the game board at each stage in the game
#  winnerdecisions : each of the decisions that the winner made
def learn_from_this(winner, boardhistory, winnerdecisions):
    print("%s won the game!" % (winner))
    print("Maybe the computer could learn from %s's experience?" % (winner))
    for idx in range(len(winnerdecisions)):
        print("\nAt the start of move %d the board looked like this:" % (idx + 1))
        print(boardhistory[idx])
        print("And %s decided to put their mark in %s" % (winner, winnerdecisions[idx]))


############################################################################
# Noughts and Crosses logic
############################################################################

# get the location of a space on the board (an index from 0 to 8)
#  using the lookup table 'deconvert'
#
#  name_of_space :  name of the space to check
#            e.g.    middle_right
def get_space_location(name_of_space):
    # uses the default spelling if found
    if name_of_space in deconvert:
        return deconvert[name_of_space]
    # otherwise tries the overrides
    return deconvert[globals()[name_of_space]]


# gets the contents of a space on the board
#
#  board         :  list of board spaces with the contents of each space
#      e.g.  [ HUMAN, COMPUTER, HUMAN, EMPTY, EMPTY, HUMAN, COMPUTER, HUMAN, COMPUTER ]
#  name_of_space :  name of the space to check
#            e.g.    middle_right
def get_space_contents(board, name_of_space):
    return board[get_space_location(name_of_space)]


# checks to see if a specific space on the board is currently empty
#
#  board         :  list of board spaces with the contents of each space
#      e.g.  [ HUMAN, COMPUTER, HUMAN, EMPTY, EMPTY, HUMAN, COMPUTER, HUMAN, COMPUTER ]
#  name_of_space :  name of the space to check
#            e.g.    middle_right
def is_space_empty(board, name_of_space):
    return get_space_contents(board, name_of_space) == EMPTY



# Creates the initial state for the game
def create_empty_board():
    debug("Creating the initial empty game board state")
    return [ EMPTY, EMPTY, EMPTY,
             EMPTY, EMPTY, EMPTY,
             EMPTY, EMPTY, EMPTY ]



# Gets the contents of the board, from the perspective of either
#  the human or the computer.
#
#  board :  list of board spaces with the current state of each space
#      e.g.  [ HUMAN, COMPUTER, HUMAN, EMPTY, EMPTY, HUMAN, COMPUTER, HUMAN, COMPUTER ]
#  who   :  either HUMAN or COMPUTER
#
# Returns the board described as PLAYER or OPPONENT
#      e.g.  [ PLAYER, OPPONENT, PLAYER, EMPTY, EMPTY, PLAYER, OPPONENT, PLAYER, OPPONENT ]
def get_board_from_perspective(board, who):
    convertedboard = []
    for move in board:
        if move == EMPTY:
            # an empty space is an empty space, from anyone's perspective
            convertedboard.append(EMPTY)
        else:
            convertedboard.append(PLAYER if move == who else OPPONENT)
    return convertedboard



############################################################################
# Noughts and Crosses user interface functions
############################################################################

# RGB colour codes
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


game_board_coordinates = {}
game_board_coordinates[top_left] = {
    "bottom_left_corner": (120, 120),
    "top_right_corner": (180, 180),
    "top_left_corner": (180, 120),
    "bottom_right_corner": (120, 180),
    "centre": (150, 150)
}
game_board_coordinates[top_middle] = {
    "bottom_left_corner": (220, 120),
    "top_right_corner": (280, 180),
    "top_left_corner": (220, 180),
    "bottom_right_corner": (280, 120),
    "centre": (250, 150)
}
game_board_coordinates[top_right] = {
    "bottom_left_corner": (320, 120),
    "top_right_corner": (380, 180),
    "top_left_corner": (320, 180),
    "bottom_right_corner": (380, 120),
    "centre": (350, 150)
}
game_board_coordinates[middle_left] = {
    "bottom_left_corner": (120, 220),
    "top_right_corner": (180, 280),
    "top_left_corner": (120, 280),
    "bottom_right_corner": (180, 220),
    "centre": (150, 250)
}
game_board_coordinates[middle_middle] = {
    "bottom_left_corner": (220, 220),
    "top_right_corner": (280, 280),
    "top_left_corner": (220, 280),
    "bottom_right_corner": (280, 220),
    "centre": (250, 250)
}
game_board_coordinates[middle_right] = {
    "bottom_left_corner": (320, 220),
    "top_right_corner": (380, 280),
    "top_left_corner": (320, 280),
    "bottom_right_corner": (380, 220),
    "centre": (350, 250)
}
game_board_coordinates[bottom_left] = {
    "bottom_left_corner": (120, 320),
    "top_right_corner": (180, 380),
    "top_left_corner": (120, 380),
    "bottom_right_corner": (180, 320),
    "centre": (150, 350)
}
game_board_coordinates[bottom_middle] = {
    "bottom_left_corner": (220, 320),
    "top_right_corner": (280, 380),
    "top_left_corner": (220, 380),
    "bottom_right_corner": (280, 320),
    "centre": (250, 350)
}
game_board_coordinates[bottom_right] = {
    "bottom_left_corner": (320, 320),
    "top_right_corner": (380, 380),
    "top_left_corner": (320, 380),
    "bottom_right_corner": (380, 320),
    "centre": (350, 350)
}


# Check if someone has won and draws a line to show the winner
#   if someone has won
#
#  who  : Who made the last move? (only need to check if they won
#          as a player who hasn't just made a move can't have won)
#         e.g.   HUMAN or COMPUTER
#
#  Returns true if someone won
#  Returns false if noone won
def display_winner(screen, board, who):
    debug("Checking if %s has won" % (who))

    gameover = False

    # we use a green line if the human wins, a red line if the computer does
    linecolour = GREEN if who == HUMAN else RED

    ######## Rows ########
    if get_space_contents(board, "top_left") == who and get_space_contents(board, "top_middle") == who and get_space_contents(board, "top_right") == who:
        pygame.draw.line(screen, linecolour, (100, 150), (400, 150), 10)
        gameover = True
    if get_space_contents(board, "middle_left") == who and get_space_contents(board, "middle_middle") == who and get_space_contents(board, "middle_right") == who:
        pygame.draw.line(screen, linecolour, (100, 250), (400, 250), 10)
        gameover = True
    if get_space_contents(board, "bottom_left") == who and get_space_contents(board, "bottom_middle") == who and get_space_contents(board, "bottom_right") == who:
        pygame.draw.line(screen, linecolour, (100, 350), (400, 350), 10)
        gameover = True

    ######## Columns ########
    if get_space_contents(board, "top_left") == who and get_space_contents(board, "middle_left") == who and get_space_contents(board, "bottom_left") == who:
        pygame.draw.line(screen, linecolour, (150, 100), (150, 400), 10)
        gameover = True
    if get_space_contents(board, "top_middle") == who and get_space_contents(board, "middle_middle") == who and get_space_contents(board, "bottom_middle") == who:
        pygame.draw.line(screen, linecolour, (250, 100), (250, 400), 10)
        gameover = True
    if get_space_contents(board, "top_right") == who and get_space_contents(board, "middle_right") == who and get_space_contents(board, "bottom_right") == who:
        pygame.draw.line(screen, linecolour, (350, 100), (350, 400), 10)
        gameover = True

    ######## Diagonals #########
    if get_space_contents(board, "top_left") == who and get_space_contents(board, "middle_middle") == who and get_space_contents(board, "bottom_right") == who:
        pygame.draw.line(screen, linecolour, (100, 100), (400, 400), 15)
        gameover = True
    if get_space_contents(board, "bottom_left") == who and get_space_contents(board, "middle_middle") == who and get_space_contents(board, "top_right") == who:
        pygame.draw.line(screen, linecolour, (400, 100), (100, 400), 15)
        gameover = True

    if gameover:
        # refresh the display if we've drawn any game-over lines
        pygame.display.update()

    return gameover



# Redraw the UI with a different background colour
#
#  board :  list of board spaces with the contents of each space
#      e.g.  [ HUMAN, COMPUTER, HUMAN, EMPTY, EMPTY, HUMAN, COMPUTER, HUMAN, COMPUTER ]
def redraw_screen(screen, colour, board):
    debug("Changing the background colour")

    # fill everything in the new background colour
    screen.fill(colour)

    # now we've covered everything, we need to redraw
    #  the game board again
    draw_game_board(screen)

    # now we need to redraw all of the moves that
    #  have been made
    for spacename in deconvert.keys():
        space_code = deconvert[spacename]

        if board[space_code] == HUMAN:
            draw_move(screen, spacename, "cross")
        elif board[space_code] == COMPUTER:
            draw_move(screen, spacename, "nought")

    # refresh now we've made changes
    pygame.display.update()



# Draw the crossed lines that make up a noughts and crosses board
def draw_game_board(screen):
    pygame.draw.rect(screen, WHITE, (195, 100, 10, 300))
    pygame.draw.rect(screen, WHITE, (295, 100, 10, 300))
    pygame.draw.rect(screen, WHITE, (100, 195, 300, 10))
    pygame.draw.rect(screen, WHITE, (100, 295, 300, 10))



# Setup the window that will be used to display the game
def prepare_game_window():
    debug("Setting up the game user interface")
    # sets up the pygame library we'll use to create the game
    pygame.init()
    # create a window that is 500 pixels wide and 500 pixels high
    screen = pygame.display.set_mode((500, 500))
    # set the title of the window
    pygame.display.set_caption("Machine Learning Noughts and Crosses")
    return screen



# Create a random RGB code to be used for the background colour
def generate_random_colour():
    debug("Generating a random colour code")
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return [r, g, b]



# Draw a new move on the game board
#
#  screen        :  The PyGame screen to draw the move on
#  name_of_space :  Name of the space to draw the move on
#                    e.g.    middle_right
#  move          :  The move to draw.
#                   It will be either "nought" or "cross"
def draw_move(screen, name_of_space, move):
    debug("Drawing a move on the game board : %s in %s" % (move, name_of_space))

    if move == "nought":
        location = game_board_coordinates[name_of_space]["centre"]
        pygame.draw.circle(screen, WHITE, location, 35 , 8)
    elif move == "cross":
        pygame.draw.line(screen, WHITE,
                         game_board_coordinates[name_of_space]["bottom_left_corner"],
                         game_board_coordinates[name_of_space]["top_right_corner"],
                         10)
        pygame.draw.line(screen, WHITE,
                         game_board_coordinates[name_of_space]["top_left_corner"],
                         game_board_coordinates[name_of_space]["bottom_right_corner"],
                         10)
    pygame.display.update()



# The user has clicked on the game board.
# Which space did they click on?
#
#  mx : the x coordinate of their click
#  my : the y coordiante of their click
#
#  Returns the name of the space they clicked on (e.g. "middle_right")
def get_click_location(mx, my):
    debug("Getting location of click in %d,%d" % (mx, my))
    if 100 < mx < 400 and 100 < my < 400:
        if my < 200:
            if mx < 200:
                return top_left
            elif mx < 300:
                return top_middle
            else:
                return top_right
        elif my < 300:
            if mx < 200:
                return middle_left
            elif mx < 300:
                return middle_middle
            else:
                return middle_right
        else:
            if mx < 200:
                return bottom_left
            elif mx < 300:
                return bottom_middle
            else:
                return bottom_right
    return "none"



# Handle a new move, by either the player or the computer
#
#  board         :  list of board spaces with the contents of each space
#      e.g.  [ HUMAN, COMPUTER, HUMAN, EMPTY, EMPTY, HUMAN, COMPUTER, HUMAN, COMPUTER ]
#  name_of_space :  name of the space the move was in
#            e.g.    middle_right
#  identity      :  whose move this is
#            e.g.    HUMAN or COMPUTER
#
#  returns true if this move ended the game
#  returns false if the game should keep going
def game_move(screen, board, name_of_space, identity):
    debug("Processing a move for %s who chose %s" % (identity, name_of_space))

    # choose the symbol for which player this is
    symbol = "cross" if identity == HUMAN else "nought"

    # draw a symbol on the board to represent the move
    draw_move(screen, name_of_space, symbol)

    # update the history of what has happened in case
    #  we want to learn from it later
    gamehistory[identity].append(board.copy())
    decisions[identity].append(name_of_space)

    # update the board to include the move
    movelocation = get_space_location(name_of_space)
    board[movelocation] = identity

    # have they won the game?
    gameover = display_winner(screen, board, identity)
    if gameover:
        # someone won! maybe an ML project could learn from this
        learn_from_this(identity, gamehistory[identity], decisions[identity])

    # the game is also over if the board is full (a draw!)
    #
    # and the board is full if both players together
    #  have made 9 moves in total
    if len(decisions[HUMAN]) + len(decisions[COMPUTER]) >= 9:
        gameover = True

    return gameover



# the machine learning model's turn
def let_computer_play(screen, board):
    computer_move = classify(board)
    print(computer_move)
    return game_move(screen, board, computer_move["class_name"], COMPUTER)




############################################################################
# Main game logic starts here
############################################################################

def debug(msg):
    # if something isn't working, uncomment the line below
    #  so you get detailed print-outs of everything that
    #  the program is doing
    # print(msg)
    pass



debug("Configuration")
debug("Using identities %s %s %s" % (EMPTY, PLAYER, OPPONENT))
debug(deconvert)

debug("Initial startup and setup")
screen = prepare_game_window()
board = create_empty_board()
redraw_screen(screen, generate_random_colour(), board)

debug("Initialising game state variables")
running = True
gameover = False

debug("Deciding who will play first")
computer_goes_first = random.choice([False, True])
if computer_goes_first:
    let_computer_play(screen, board)


while running:
    # wait for the user to do something...
    event = pygame.event.wait()

    if event.type == pygame.QUIT:
        running = False

    if event.type == pygame.MOUSEBUTTONDOWN and gameover == False:
        # what has the user clicked on?
        mx, my = pygame.mouse.get_pos()
        location_name = get_click_location(mx, my)

        if location_name == "none":
            # user clicked on none of the spaces so we'll
            #  change the colour for them instead!
            redraw_screen(screen, generate_random_colour(), board)

        elif is_space_empty(board, location_name):
            # the user clicked on an empty space
            gameover = game_move(screen, board, location_name, HUMAN)

            # if we're still going, it is the computer's turn next
            if gameover == False:
                # the computer chooses where to play
                gameover = let_computer_play(screen, board)

        # ignore anything else the user clicked on while we
        #  were processing their click, so they don't try to
        #  sneakily have lots of moves at once
        pygame.event.clear()

# explicitly quit pygame to ensure the app terminates correctly
#  cf. https://www.pygame.org/wiki/FrequentlyAskedQuestions
pygame.quit()'''

'''from sklearn.preprocessing import OneHotEncoder
import numpy as np


words = np.array([["사과"], ["바나나"], ["포도"]])
enoder = OneHotEncoder(sparse_output = False)
onehot = enoder.fit_transform(words)
print(onehot)

from sklearn.feature_extraction.text import CountVectorizer
corpus = ["나는 고양이를 좋아해", "나는 강아지를 좋아해", "고양이는 귀엽다"]
vect = CountVectorizer()
X = vect.fit_transform(corpus)
print(vect.get_feature_names_out())
print(X.toarray())'''
'''
import gensim.downloader as api

model = api.load("glove-wiki-gigaword-50")

similar_words = model.most_similar("dog", topn=5)
print(similar_words)

s1 = model.similarity("dog", "bear")
s2 = model.similarity("dog", "car")

print("dog vs bear:", s1)
print("dog vs car:", s2)'''
'''
from gensim.models import Word2Vec

corpus = [
    ["고양이", "는", "동물", "이다"],
    ["고양이", "는", "생선", "을", "좋아한다"],
    ["강아지", "는", "동물", "이다"],
    ["강아지", "는", "뼈", "를", "좋아한다"],
    ["고양이", "와", "강아지", "는", "귀엽다"],
    ["사과", "는", "과일", "이다"],
    ["바나나", "는", "과일", "이다"],
    ["포도", "는", "과일", "이다"],
    ["사과", "와", "바나나", "는", "맛있다"],
]

w2v = Word2Vec(
    sentences=corpus,
    vector_size=50,
    window=2,
    min_count=1,
    sg=1,
    negative=5,
    epochs=200,
    seed=42
)

print("유사도(고양이, 강아지):", w2v.wv.similarity("고양이", "강아지"))
print("유사도(사과, 바나나):", w2v.wv.similarity("사과", "바나나"))

print("\n'고양이'와 비슷한 단어 top-5:")
for w, s in w2v.wv.most_similar("고양이", topn=5):
    print(f"{w}: {s:.3f}")'''
    
import random
import gensim.downloader as api

KO2EN = {
    "강아지":"dog","개":"dog","고양이":"cat","곰":"bear","호랑이":"tiger","사자":"lion",
    "코끼리":"elephant","말":"horse","늑대":"wolf","여우":"fox","토끼":"rabbit","사슴":"deer",
    "원숭이":"monkey","침팬지":"chimpanzee","고릴라":"gorilla","판다":"panda","소":"cow",
    "돼지":"pig","양":"sheep","염소":"goat","닭":"chicken","독수리":"eagle","오리":"duck"
}

"""
KO2EN = {
    "사과":"apple","바나나":"banana","포도":"grape","오렌지":"orange","복숭아":"peach",
    "딸기":"strawberry","수박":"watermelon","레몬":"lemon","파인애플":"pineapple","토마토":"tomato",
    "감자":"potato","양파":"onion","당근":"carrot","양배추":"cabbage","상추":"lettuce",
    "오이":"cucumber","쌀":"rice","빵":"bread","치즈":"cheese","버터":"butter",
    "우유":"milk","달걀":"egg","치킨":"chicken","소고기":"beef","돼지고기":"pork",
    "생선":"fish","수프":"soup","샐러드":"salad","피자":"pizza","햄버거":"hamburger"
}
KO2EN = {
    "전화기":"phone","컴퓨터":"computer","노트북":"laptop","책":"book","의자":"chair","테이블":"table","책상":"desk",
    "병":"bottle","컵":"cup","유리컵":"glass","숟가락":"spoon","포크":"fork","칼":"knife","접시":"plate",
    "가방":"bag","배낭":"backpack","지갑":"wallet","시계":"watch","벽시계":"clock","카메라":"camera",
    "텔레비전":"television","라디오":"radio","램프":"lamp","문":"door","창문":"window","열쇠":"key",
    "우산":"umbrella","신발":"shoes","모자":"hat","수건":"towel","비누":"soap","거울":"mirror",
    "침대":"bed","소파":"sofa","베개":"pillow","담요":"blanket","마우스":"mouse","키보드":"keyboard",
    "자전거":"bicycle","손전등":"flashlight","스마트폰":"smartphone","프린터":"printer","리모컨":"remote",
    "선풍기":"fan","에어컨":"air","가위":"scissors","테이프":"tape","지우개":"eraser","연필":"pencil","펜":"pen",
    "종이":"paper","상자":"box","칫솔":"toothbrush","치약":"toothpaste","물병":"water bottle"
}
"""'''
EN2KO = {v:k for k,v in KO2EN.items()}
print("임베딩 모델 로딩 중... ")
model = api.load("glove-wiki-gigaword-50")
vocab = set(model.key_to_index.keys())

ANIMALS_KO = [ko for ko,en in KO2EN.items() if en in vocab]
MAX_TURNS = 7

def play():
    answer_ko = random.choice(ANIMALS_KO) 
    answer_en = KO2EN[answer_ko]

    print("=== 동물 유사도 퀴즈 ===")
    print("아래 동물들 중 하나가 정답입니다. 맞춰보세요!\n")
    print(", ".join(ANIMALS_KO))
    print(f"\n시도 기회: {MAX_TURNS}번\n")

    for turn in range(1, MAX_TURNS + 1):
        guess_ko = input(f"[{turn}/{MAX_TURNS}] 추측 동물(한글만): ").strip()

        if guess_ko not in ANIMALS_KO:
            print("  - 목록에 없는 동물입니다. 후보 중에서 골라주세요!")
            continue
        if guess_ko == answer_ko:
            print(f"정답! {answer_ko} ({answer_en})")
            return

        guess_en = KO2EN[guess_ko]
        sim = model.similarity(guess_en, answer_en)
        print(f"  - 유사도: {sim:.3f}")
        near = [EN2KO[w] for w,_ in model.most_similar(answer_en, topn=30) 
                if w in EN2KO and EN2KO[w] in ANIMALS_KO and w != answer_en][:3]
        if near:
            print("힌트:", ", ".join(near))
    print(f"\n실패! 정답은 {answer_ko} ({answer_en})")

play()'''
'''
import pandas as pd
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
import re, os, pickle, urllib.request
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
train_data = pd.read_table('ratings.txt')
TOKEN_FILE = "tokenized.pkl"

print('리뷰 개수 :',len(train_data))
print(train_data.head())

train_data = train_data.dropna(how="any").reset_index(drop=True)
print(len(train_data))
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True)
train_data["document"] = train_data["document"].str.replace(r"\s+", " ", regex=True).str.strip()

okt = Okt()
stopwords = {
    "의","가","이","은","들","는","좀","잘","걍","과","도","를","으로","자","에","와","한","하다",
    "에서","에게","께서","보다","부터","까지","처럼","이라","였다","되다","이다"
}
tokenized_data = []

def tokenize(sent: str):
    if not isinstance(sent, str) or not sent:
        return []
    toks = okt.morphs(sent, norm=True, stem=True)
    return [t for t in toks if t not in stopwords and len(t) > 1]

print("[토큰화] 진행 중...")
tokenized = [tokenize(s) for s in tqdm(train_data["document"], total=len(train_data), desc="Tokenizing", ncols=80)]
labels = train_data["label"].astype(int).to_numpy()

# 4) 저장
with open(TOKEN_FILE, "wb") as f:
    pickle.dump({"tokenized": tokenized, "labels": labels}, f)

print(f"[저장 완료] {TOKEN_FILE}")

for i in range(3):
    print(f"\n원문: {train_data['document'][i]}")
    print(f"토큰화: {tokenized[i]}")
    print(f"레이블: {labels[i]}")'''
'''
import pickle
import numpy as np
from tqdm.auto import tqdm
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from konlpy.tag import Okt

with open("tokenized.pkl", "rb") as f:
    data = pickle.load(f)

tokenized = data["tokenized"]
labels = data["labels"]

print("\n[Word2Vec] 학습 시작")
w2v = Word2Vec(
    sentences=tokenized,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=1,
    negative=5,
    epochs=10,
    seed=42
)
print("[Word2Vec] 어휘 수:", len(w2v.wv))

dim = w2v.vector_size
print("\n[문장 벡터] 평균 임베딩 생성")

X = []
for toks in tqdm(tokenized, total=len(tokenized), desc="Averaging", ncols=80):
    vecs = [w2v.wv[w] for w in toks if w in w2v.wv]
    X.append(np.mean(vecs, axis=0) if vecs else np.zeros(dim))

X = np.array(X)
y = labels


X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=2000, solver="lbfgs").fit(X_tr, y_tr)

pred = clf.predict(X_te)
print("\nAccuracy:", accuracy_score(y_te, pred))

okt = Okt()
stopwords = {
    "의","가","이","은","들","는","좀","잘","걍","과","도","를","으로","자","에","와","한","하다",
    "에서","에게","께서","보다","부터","까지","처럼","이라","였다","되다","이다"
}
def tokenize(sent: str):
    if not isinstance(sent, str) or not sent:
        return []
    toks = okt.morphs(sent, norm=True, stem=True)
    return [t for t in toks if t not in stopwords and len(t) > 1]

def predict_sentiment(text: str):
    toks = tokenize(text)
    vecs = [w2v.wv[w] for w in toks if w in w2v.wv]
    v = np.mean(vecs, axis=0).reshape(1, -1) if vecs else np.zeros(dim).reshape(1, -1)
    proba = clf.predict_proba(v)[0]
    label = clf.predict(v)[0]
    return {"label": int(label), "proba_neg": float(proba[0]), "proba_pos": float(proba[1]), "tokens": toks}

print(predict_sentiment("진짜 감동적입니다. "))'''
'''
import streamlit as st

st.title("Streamlit 시작")
st.write("안녕하세요! 첫 번째 예제입니다.")

number = st.number_input("숫자를 입력하세요", min_value=0, max_value=100, value=10)

if st.button("두 배로 계산하기"):
    st.write(f"👉 결과: {number * 2}")

name = st.text_input("이름을 입력하세요", "")
if name:
    st.write(f"반가워요, **{name}** 님!")'''
'''
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model():
    model_name = "skt/kogpt2-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

def generate_text(prompt, max_len=40):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_len, do_sample=True, top_p=0.9, temperature=0.8)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)'''
'''
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="인공지능은 생각을 할 수 있나?"
)
print(response.text)'''
'''
import os
from dotenv import load_dotenv
import streamlit as st
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

st.set_page_config(page_title="답변 생성기", page_icon="🤖")
st.title("🤖 Gemini 답변 생성 웹앱")

if "question" not in st.session_state:
    st.session_state.question = "인공지능은 생각을 할 수 있나?"
if "answer" not in st.session_state:
    st.session_state.answer = ""

# 질문 입력 (세션 상태와 연결)
st.session_state.question = st.text_area("질문 입력", st.session_state.question)

# 버튼들 가로로 배치
col1, col2 = st.columns(2)

with col1:
    if st.button("답변 받기"):
        try:
            with st.spinner("모델이 답변 중…"):
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=st.session_state.question
                )
                # 결과 저장
            st.session_state.answer = response.text

        except Exception as e:
            st.error(f"오류: {e}")

with col2:
   if st.button("초기화"):
        st.session_state.question = ""
        st.session_state.answer = ""
st.subheader("답변")
st.write(st.session_state.answer)'''
'''
from openai import OpenAI
import base64
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
prompt = "귀여운 흰 아기고양이"

response = client.images.generate(
    model="dall-e-3",
    prompt=prompt,
    size="1024x1024",
    n=1,
    response_format="b64_json"  # base64로 받기
)

image_base64 = response.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

with open("generated_image.png", "wb") as f:
    f.write(image_bytes)

print("이미지 생성 완료!")'''


from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = "너는 파이썬 개발자야. 코딩에 관련된 질문에 친절하게 대답하지."

history = []
print("Gemini 챗봇이 시작되었습니다. 'quit'을 입력하면 종료됩니다.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("대화를 종료합니다.")
        break

    history_text = ""
    for m in history:
        role = "사용자" if m["role"] == "user" else "AI"
        history_text += f"{role}: {m['content']}\n"

    contents = (
        f"[시스템]\n{SYSTEM_PROMPT}\n\n"
        f"[대화]\n{history_text}사용자: {user_input}\nAI:"
    )

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.7, 
        ),
    )

    answer = getattr(resp, "text", "") or "(빈 응답)"
    print("AI:", answer, "\n")

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": answer})