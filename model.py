from roboflow import Roboflow
import numpy as np

rf = Roboflow(api_key="Yr1jfwkQBJcCCKmwpgwI")
project = rf.workspace().project("chess-corners-wzsil")
model_board = project.version(1).model

mod = model_board.predict(r"C:\Users\heetm\OneDrive\Desktop\Chess_board\try.jpg", confidence=25, overlap=50).json()
# infer on a local image
#print(mod)
#get the x and y coordinates from the jsom file

boxes = np.array([0, 0])
for bounding_box in mod['predictions']:
    x = bounding_box['x'] - 100
    y = bounding_box['y'] - 100
    box = np.array([x, y])
    #print(box)
    boxes = np.vstack((boxes, box))

#remove the first row of zeros
boxes = np.delete(boxes, 0, 0)
print(boxes)

# visualize your prediction
model_board.predict(r"C:\Users\heetm\OneDrive\Desktop\Chess_board\try.jpg", confidence=25, overlap=50).save("prediction.jpg")
#mod.save("prediction.jpg")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('prediction.jpg')
imgplot = plt.imshow(img)
plt.show()