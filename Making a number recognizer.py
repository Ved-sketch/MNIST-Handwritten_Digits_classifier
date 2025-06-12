from sklearn.datasets import fetch_openml
import numpy as np
import joblib
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mnist = fetch_openml('mnist_784')
x = mnist['data']
y = mnist['target']

# doing the train-test splitting
x_train = x[:60000]
x_test = x[60000:]

y_train = y[:60000]
y_test = y[60000:]

# shuffling
shuffling_training = np.random.permutation(60000)
x_train = x_train.iloc[shuffling_training]
y_train = y_train.iloc[shuffling_training]


# Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(x_train_scaled, y_train)

# testing the model
predictions = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test,predictions)
print("Accuracy",accuracy)

# Making the function for testing custom inputs
def recognize(image_path):
    img = Image.open(image_path).convert('L') # L means grayscale
    img = img.resize((28,28)) # resizing the image to meet the format of mnist
    img_array = np.array(img) # converting to an array
    img_array = 255 - img_array # color inversion
    img_flat = img_array.reshape(1,-1)
    img_flat_scaled = scaler.transform(img_flat)

    # returning predictions
    prediction = model.predict(img_flat_scaled)
    print("Predicted Number:",prediction[0])

# Saving the model
joblib.dump(model,"Handwritten_digit_recognizer.pkl")

#recognize("one.png")
#recognize("two.png")
#recognize("three.png")
#recognize("four.png")
#recognize("six.png")
#recognize("number.png")

# model was not working initially. why? because i was not passing images of resolution 28x28.
# now i have corrected it and it is giving decent predictions. This resolution thing is a point to be noted.
