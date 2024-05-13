from roboflow import Roboflow
import numpy as np

rf = Roboflow(api_key="ExPpgdtw6WZeTqCWsASd")
project = rf.workspace().project("chess-corners-rvw38")
model_board = project.version(1).model

mod = model_board.predict(r"C:\Users\heetm\OneDrive\Desktop\DL Assignment\Code\chesscog\data\render\test\0093.png", confidence=5, overlap=30).json()
# infer on a local image
print(mod)
#get the x and y coordinates from the jsom file
boxes = np.array([0, 0])
for bounding_box in mod['predictions']:
    x = bounding_box['x']
    y = bounding_box['y']
    #convert thre x and y coordinates to numpy array
    box = np.array([x, y])
    print(box)
    #can you combine all the boxes in a single numpy array but all as a single row
    boxes = np.vstack((boxes, box))

#remove the first row of zeros
boxes = np.delete(boxes, 0, 0)
print(boxes)
print(type(boxes))


#save the model as .pt file
#model.save("model.pt")

# visualize your prediction
model_board.predict(r"C:\Users\heetm\OneDrive\Desktop\DL Assignment\Code\chesscog\data\render\test\0093.png", confidence=5, overlap=30).save("prediction.jpg")

# example box object from the Pillow library


# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('prediction.jpg')
imgplot = plt.imshow(img)
plt.show()