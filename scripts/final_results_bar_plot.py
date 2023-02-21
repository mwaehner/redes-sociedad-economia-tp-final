import numpy as np
import matplotlib.pyplot as plt

# creating the dataset
data = {'Hidden_10': 90.6, 'Hidden_30++': 91.6, 'Hidden_30_dropout': 89.5}

courses = list(data.keys())
values = list(data.values())

fig = plt.figure(figsize=(8, 5))

# creating the bar plot
plt.bar(courses, values, color='maroon',
        width=0.4)

plt.ylim([85, 92])

plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Mejor f1-score para la clase de portugués")
plt.show()