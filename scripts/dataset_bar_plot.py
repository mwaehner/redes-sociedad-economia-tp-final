import os

import numpy as np
import matplotlib.pyplot as plt

# set width of bar
import pandas as pd

df_math = pd.read_csv(
    "..\\student-mat.csv",
    sep=",",
    usecols=['G3']
)

df_por = pd.read_csv(
    "..\\student-por.csv",
    sep=",",
    usecols=['G3']
)

barWidth = 0.4
fig = plt.subplots(figsize=(9, 8))

students_passed_math = len(df_math[df_math['G3'] >= 10])
students_passed_por = len(df_por[df_por['G3'] >= 10])
students_not_passed_math = len(df_math) - students_passed_math
students_not_passed_por = len(df_por) - students_passed_por
# set height of bar
PASSED = [students_passed_math, students_passed_por]
NOT_PASSED = [students_not_passed_math, students_not_passed_por]

# Set posPASSEDion of bar on X axis
br1 = np.arange(len(NOT_PASSED))
br2 = [x + barWidth for x in br1]

# Make the plot

plt.bar(br1, NOT_PASSED, color='r', width=barWidth,
        edgecolor='grey', label='No aprobado')
plt.bar(br2, PASSED, color='g', width=barWidth,
        edgecolor='grey', label='Aprobado')

# Adding Xticks
plt.ylabel('Cantidad de estudiantes', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(NOT_PASSED))],
           ['Matemática', 'Portugués'])

plt.legend()
plt.show()