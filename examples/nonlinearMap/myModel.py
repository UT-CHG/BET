# Copyright (C) 2016 The BET Development Team

# -*- coding: utf-8 -*-
import numpy as np
import math as m

'''
Suggested changes for user:

Try setting QoI_num = 2.

Play around with the x1, y1, and/or, x2, y2 values to try and
"optimize" the QoI to give the highest probability region
on the reference parameter above.

Hint: Try using QoI_num = 1 and systematically varying the
x1 and y1 values to find QoI with contour structures (as inferred
through the 2D marginal plots) that are nearly orthogonal.

Some interesting pairs of QoI to compare are:
(x1,y1)=(0.5,0.5) and (x2,y2)=(0.25,0.25)
(x1,y1)=(0.5,0.5) and (x2,y2)=(0.15,0.15)
(x1,y1)=(0.5,0.5) and (x2,y2)=(0.25,0.15)
'''
# Choose the number of QoI
QoI_num = 1

# Specify the spatial points to take measurements of solution defining the QoI
if QoI_num == 1:
    x1 = 0.5
    y1 = 0.5
    x = np.array([x1])
    y = np.array([y1])
else:
    x1 = 0.5
    y1 = 0.15
    x2 = 0.15
    y2 = 0.25
    x = np.array([x1, x2])
    y = np.array([y1, y2])

class QoI_component(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def eval(self, parameter_samples):
        if parameter_samples.shape == (2,):
            lam1 = parameter_samples[0]
            lam2 = parameter_samples[1]
        else:
            lam1 = parameter_samples[:,0]
            lam2 = parameter_samples[:,1]
        z = np.sin(m.pi * self.x * lam1) * np.sin(m.pi * self.y * lam2)
        return z

# Specify the QoI maps
if QoI_num == 1:
    def QoI_map(parameter_samples):
        Q1 = QoI_component(x[0], y[0])
        return np.array([Q1.eval(parameter_samples)]).transpose()
else:
    def QoI_map(parameter_samples):
        Q1 = QoI_component(x[0], y[0])
        Q2 = QoI_component(x[1], y[1])
        return np.array([Q1.eval(parameter_samples), Q2.eval(parameter_samples)]).transpose()

# Define a model that is the QoI map
def my_model(parameter_samples):
    QoI_samples = QoI_map(parameter_samples)
    return QoI_samples
