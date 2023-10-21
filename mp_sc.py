import cv2
import csv
import mediapipe as mp
import numpy as np
import glob
from tqdm import tqdm

def insert_data_to_csv(img_path, points,num, csv_path):
    # Insert image directory and number into CSV file
    # print("into csv")
    with open(csv_path, mode='a') as csv_file:
        writer = csv.writer(csv_file)
        # writer.writerow([img_path, points, num, len(points)])
        writer.writerow([img_path, points, num])
    # print(num)
    # print("into csv done")

def landmarks(image):
    # print("landmarks")
    # Create a mp.Hands object to detect hand landmarks
    # image = np.array(image)
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)

    # Load the image and convert it to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to detect hand landmarks
    results = hands.process(image)
    coords = []
    counter = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            counter+=1
            mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            
            for c in hand_landmarks.landmark:
                coords.append(np.ceil(c.x * image.shape[1]))
                coords.append(np.ceil(c.y * image.shape[0]))
    
    # img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # name = "imgs/landmarks_img" + str(num) + ".jpg"
    # cv2.imwrite(name, img)

    # print("landmarks done")
    counter = 0

    return coords, image


# csv_path = "data.csv"

# image_files = []
# path = 'C:\\Northeastern_University\\Sem_2\\RSS\\Project\\data_images\\'
# hand_s = input("sign: ") + "\\*.jpg"
# path = path + hand_s
# y_class = int(input("class type: "))

# for filename in glob.glob(path):
#     image_files.append(filename)

# for idx, file in tqdm(enumerate(image_files)):
#     #use frame as image here to get points, call your function here
#     points = landmarks(cv2.imread(file),idx)
#     # Insert data into CSV file
#     # insert_data_to_csv(file,points,y_class, csv_path)
