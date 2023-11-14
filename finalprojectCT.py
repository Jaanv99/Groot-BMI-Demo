# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 15:20:30 2023

@author: jahna
"""

import os
import sys
import math
import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.signal import find_peaks
from sklearn.metrics import r2_score

qtcreator_file  = "CFA.ui" # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.BrowseButton.clicked.connect(self.openFile)
        self.plotButton.clicked.connect(self.plot)
        self.ClearButton.clicked.connect(self.clear)
        self.ExitButton.clicked.connect(self.closeprog)
        self.AddButton.clicked.connect(self.AddPoints)
        self.SaveButton.clicked.connect(self.savepng)
     
    def AddPoints(self):
        points = self.xyinput.toPlainText()
        self.x = []
        self.y = []
        for line in points.split('\n'):
            l = line.strip()
            xy = l.split(',')
            xp = float(xy[0].strip())
            yp = float(xy[1].strip())
            self.x.append(xp)
            self.y.append(yp)
        x_series = round(pd.Series(self.x, name='X'),2)
        y_series = round(pd.Series(self.y, name='Y'),2)
        result = pd.concat([x_series, y_series], axis=1)
        self.Points.setText(result.to_string(index=False))
        pointslist = list(zip(self.x, self.y))
        # Sorting the x values in ascending order
        sorted_points = sorted(pointslist, key=lambda point: point[0])
        # Converting to lists
        sorted_x, sorted_y = zip(*sorted_points)
        self.x = list(sorted_x)
        self.y = list(sorted_y)
        return self.x, self.y
        
    def openFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Excel File', '', 'Excel Files (*.xlsx *.xls)', options=options)
        excel_data = pd.read_excel(file_name, header=None)
        self.x_col = excel_data.iloc[:, 0]
        self.y_col = excel_data.iloc[:, 1]
        pointslist = list(zip(self.x_col, self.y_col))
        # Sorting the x values in ascending order
        sorted_points = sorted(pointslist, key=lambda point: point[0])
        # Converting to lists
        sorted_x, sorted_y = zip(*sorted_points)
        self.x = list(sorted_x)
        self.y = list(sorted_y)
        first_two_columns = round(excel_data.iloc[:, :2],2)
        self.Points.setText(first_two_columns.to_string(index=False, header=None))
        return self.x, self.y, self.x_col, self.y_col

    def plot(self):
        # Creating a Matplotlib figure and canvas
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        # Adding the Matplotlib canvas to the existing QGraphicsView
        self.scene = QtWidgets.QGraphicsScene(self)
        self.scene.addWidget(self.canvas)
        self.graphicsView.setScene(self.scene)
        # Creating a single subplot during initialization
        self.ax = self.fig.add_subplot(111)
        # Clear the previous scatter plot
        self.ax.clear()
        # Adding the new scatter plot
        self.ax.scatter(self.x, self.y)
        # Add curve fitting data here
        # Make the x and y sets into numpy arrays:
        x_arr = np.array(self.x)
        y_arr = np.array(self.y)
        # Type of Boundary Condition
        BC = self.BCspinBox.value()
        order = self.OrderSpinBox.value()
        custom = self.textequation.toPlainText()
        if custom == '':
            # PLot based on BC type and for order of equation input number
            if BC == 1:
                print(BC)
                # Convert them to a pandas dataframe for calculating the moving average - used the pandas rolling.mean() approach because it returns the same number of values as there were at the start (good for plotting the same x values) since the function takes averages over a number of points so some points get "lost" - those points are replaces with NaN points (not a number).
                data = pd.DataFrame({'x': x_arr, 'y': y_arr})
        
                # Apply moving average using average data rolling from pandas - two averages for better smoothing, one with a window of 5 points and one with an average of 8 - bigger window makes it smoother but offsets it more.
                data_smooth = data.rolling(window = 5).mean()
                data_smooth2 = data_smooth.rolling(window = 8).mean()
        
                # Find the maximum point - the boundary condition - in the smoothed data set, and plot it to check.
                max_id = list(data_smooth2[['y']].idxmax())[0]
                print(max_id)
                
                # Split the original data set using the new max_id index:    
                x_1 = self.x[:max_id].copy()
                x_2 = self.x[max_id:].copy()
                y_1 = self.y[:max_id].copy()
                y_2 = self.y[max_id:].copy()
                
                xp_1 = np.linspace(min(x_1), max(x_1), 50)
                xp_2 = np.linspace(min(x_2), max(x_2), 50)
                
                # Curve fit:
                poly1_1 = np.polyfit(x_1, y_1, deg = order)
                poly1_2 = np.polyfit(x_2, y_2, deg = order)
                
                predict1_1 = np.poly1d(poly1_1)
                predict1_2 = np.poly1d(poly1_2)
                          
                self.ax.plot(xp_1, predict1_1(xp_1), 'r-', label = 'y_1')
                self.ax.plot(xp_2, predict1_2(xp_2), 'b-', label = 'y_2')
                self.ax.legend()
                
                r2_1_l = r2_score(y_1, predict1_1(x_1))
                r2_1_r = r2_score(y_2, predict1_2(x_2))
                
                n = order
                if n == 1:
                    if poly1_1[n] > 0:
                        # Print statement for console
                        equation_str1 = 'y_1 = {0:.5f}x + {1:.5f}'.format(poly1_1[0], poly1_1[n])
                        equation_str2 = 'y_2 = {0:.5f}x + {1:.5f}'.format(poly1_2[0], poly1_2[n])
                    else:
                        equation_str1 = 'y_1 = {0:.5f}x - {1:.5f}'.format(poly1_1[0], abs(poly1_1[n]))
                        equation_str2 = 'y_2 = {0:.5f}x - {1:.5f}'.format(poly1_2[0], abs(poly1_2[n]))
                    equation_str1 += f" (r2 = {r2_1_l:.5f})"
                    equation_str2 += f" (r2 = {r2_1_r:.5f})"
                    self.printeqnR2.setText(equation_str1 + '\n' + equation_str2)
                else: # For higher order equations
                    equation_str1 = 'y_1 = {0:.5f}x^{1}'.format(poly1_1[0], n)
                    equation_str2 = 'y_2 = {0:.5f}x^{1}'.format(poly1_2[0], n)
                    for j in range(1, n + 1):
                        coefficient1 = poly1_1[j]
                        power1 = n-j
                        if coefficient1 < 0:
                            sign = "-"
                        else:
                            sign = "+"
                        if power1 == 0:
                            # Appending the term to the equation string
                            equation_str1 += f" {sign} {abs(coefficient1):.5f}"
                        elif power1 == 1:
                            equation_str1 += f" {sign} {abs(coefficient1):.5f}x"
                        else:
                            equation_str1 += f" {sign} {abs(coefficient1):.5f}x^{power1}"
                    equation_str1 += f" (r2 = {r2_1_l:.5f})"
                    for j in range(1, n + 1):
                        coefficient2 = poly1_2[j]
                        power2 = n-j
                        if coefficient2 < 0:
                            sign = "-"
                        else:
                            sign = "+"
                        if power2 == 0:
                            # Appending the term to the equation string
                            equation_str2 += f" {sign} {abs(coefficient2):.5f}"
                        elif power2 == 1:
                            equation_str2 += f" {sign} {abs(coefficient2):.5f}x"
                        else:
                            equation_str2 += f" {sign} {abs(coefficient2):.5f}x^{power2}"
                    equation_str2 += f" (r2 = {r2_1_r:.5f})"
                self.printeqnR2.setText(equation_str1 + '\n' + equation_str2)
                
            elif BC == 2:
                print(BC)
                # Calculate the moving averages of every 4 points
                window_size = 4
                moving_averages = np.convolve(y_arr, np.ones(window_size)/window_size, mode='valid')
                
                # Calculate errors as the differences between original data and moving averages
                errors = y_arr[window_size-1:] - moving_averages
                
                TV = self.TValue.toPlainText()
                if not TV == '':  # Check if TV input exists
                    threshold = float(TV)
                else:
                    # Set threshold dynamically based on the standard deviation of errors
                    threshold = 2*(np.std(errors))
                print(threshold)
                
                # Find indices where the error exceeds the threshold
                jump_indices = np.where(np.abs(errors) > threshold)[0] + window_size - 1
            
                # Print the detected jump indices
                print("Detected jump indices:", jump_indices)
        
                if len(jump_indices) > 2:
                    max_id = jump_indices[0]
                    next_id = jump_indices[2]
                    # Split the original data set using the new max_id index:    
                    x_1 = self.x[:max_id].copy()
                    x_2 = self.x[max_id:next_id].copy()
                    x_3 = self.x[next_id:].copy()
            
                    y_1 = self.y[:max_id].copy()
                    y_2 = self.y[max_id:next_id].copy()
                    y_3 = self.y[next_id:].copy()
                    
                    xp_1 = np.linspace(min(x_1), max(x_1), 50)
                    xp_2 = np.linspace(min(x_2), max(x_2), 50)
                    xp_3 = np.linspace(min(x_3), max(x_3), 50)
                                   
                    # Curve fit:
                    poly1_1 = np.polyfit(x_1, y_1, deg = order)
                    poly1_2 = np.polyfit(x_2, y_2, deg = order)
                    poly1_3 = np.polyfit(x_3, y_3, deg = order)
                    
                    predict1_1 = np.poly1d(poly1_1)
                    predict1_2 = np.poly1d(poly1_2)
                    predict1_3 = np.poly1d(poly1_3)
                    self.ax.plot(xp_1, predict1_1(xp_1), 'r-', label = 'Fit 1')
                    self.ax.plot(xp_2, predict1_2(xp_2), 'b-', label = 'Fit 2')
                    self.ax.plot(xp_3, predict1_3(xp_3), 'g-', label = 'Fit 3')
                    self.ax.legend()
                    
                    r2_1 = r2_score(y_1, predict1_1(x_1))
                    r2_2 = r2_score(y_2, predict1_2(x_2))
                    r2_3 = r2_score(y_2, predict1_2(x_2))
                
                    n = order
                    if n == 1:
                        if poly1_1[n] > 0:
                            # Print statement for console
                            equation_str1 = 'y_1 = {0:.5f}x + {1:.5f}'.format(poly1_1[0], poly1_1[n])
                            equation_str2 = 'y_2 = {0:.5f}x + {1:.5f}'.format(poly1_2[0], poly1_2[n])
                            equation_str3 = 'y_3 = {0:.5f}x + {1:.5f}'.format(poly1_3[0], poly1_3[n])
                        else:
                            equation_str1 = 'y_1 = {0:.5f}x - {1:.5f}'.format(poly1_1[0], abs(poly1_1[n]))
                            equation_str2 = 'y_2 = {0:.5f}x - {1:.5f}'.format(poly1_2[0], abs(poly1_2[n]))
                            equation_str3 = 'y_3 = {0:.5f}x + {1:.5f}'.format(poly1_3[0], abs(poly1_3[n]))
                        equation_str1 += f" (r2 = {r2_1:.5f})"
                        equation_str2 += f" (r2 = {r2_2:.5f})"
                        equation_str3 += f" (r2 = {r2_3:.5f})"
                        self.printeqnR2.setText(equation_str1 + '\n' + equation_str2 + '\n' + equation_str3)
                    else: # For higher order equations
                        equation_str1 = 'y_1 = {0:.5f}x^{1}'.format(poly1_1[0], n)
                        equation_str2 = 'y_2 = {0:.5f}x^{1}'.format(poly1_2[0], n)
                        equation_str3 = 'y_3 = {0:.5f}x^{1}'.format(poly1_3[0], n)
                        for j in range(1, n + 1):
                            coefficient1 = poly1_1[j]
                            power1 = n-j
                            if coefficient1 < 0:
                                sign = "-"
                            else:
                                sign = "+"
                            if power1 == 0:
                                # Appending the term to the equation string
                                equation_str1 += f" {sign} {abs(coefficient1):.5f}"
                            elif power1 == 1:
                                equation_str1 += f" {sign} {abs(coefficient1):.5f}x"
                            else:
                                equation_str1 += f" {sign} {abs(coefficient1):.5f}x^{power1}"
                        equation_str1 += f" (r2 = {r2_1:.5f})"
                        for j in range(1, n + 1):
                            coefficient2 = poly1_2[j]
                            power2 = n-j
                            if coefficient2 < 0:
                                sign = "-"
                            else:
                                sign = "+"
                            if power2 == 0:
                                # Appending the term to the equation string
                                equation_str2 += f" {sign} {abs(coefficient2):.5f}"
                            elif power2 == 1:
                                equation_str2 += f" {sign} {abs(coefficient2):.5f}x"
                            else:
                                equation_str2 += f" {sign} {abs(coefficient2):.5f}x^{power2}"
                        equation_str2 += f" (r2 = {r2_2:.5f})"
                        for j in range(1, n + 1):
                            coefficient3 = poly1_3[j]
                            power3 = n-j
                            if coefficient3 < 0:
                                sign = "-"
                            else:
                                sign = "+"
                            if power3 == 0:
                                # Appending the term to the equation string
                                equation_str3 += f" {sign} {abs(coefficient3):.5f}"
                            elif power3 == 1:
                                equation_str3 += f" {sign} {abs(coefficient3):.5f}x"
                            else:
                                equation_str3 += f" {sign} {abs(coefficient3):.5f}x^{power3}"
                        equation_str3 += f" (r2 = {r2_3:.5f})"
                    self.printeqnR2.setText(equation_str1 + '\n' + equation_str2 + '\n' + equation_str3)
                
                elif len(jump_indices) > 0 and len(jump_indices) < 3:
                        max_id = jump_indices[0]
                        # Split the original data set using the new max_id index:    
                        x_1 = self.x[:max_id].copy()
                        x_2 = self.x[max_id:].copy()
                
                        y_1 = self.y[:max_id].copy()
                        y_2 = self.y[max_id:].copy()
                        
                        xp_1 = np.linspace(min(x_1), max(x_1), 50)
                        xp_2 = np.linspace(min(x_2), max(x_2), 50)
                        
                        # Curve fit:
                        poly1_1 = np.polyfit(x_1, y_1, order)
                        poly1_2 = np.polyfit(x_2, y_2, deg = order)
                        
                        predict1_1 = np.poly1d(poly1_1)
                        predict1_2 = np.poly1d(poly1_2)
                        self.ax.plot(xp_1, predict1_1(xp_1), 'r-', label = 'Fit 1')
                        self.ax.plot(xp_2, predict1_2(xp_2), 'b-', label = 'Fit 2')
                        self.ax.legend()
                        
                        r2_1_l = r2_score(y_1, predict1_1(x_1))
                        r2_1_r = r2_score(y_2, predict1_2(x_2))
                        
                        n = order
                        if n == 1:
                            if poly1_1[n] > 0:
                                # Print statement for console
                                equation_str1 = 'y_1 = {0:.5f}x + {1:.5f}'.format(poly1_1[0], poly1_1[n])
                                equation_str2 = 'y_2 = {0:.5f}x + {1:.5f}'.format(poly1_2[0], poly1_2[n])
                            else:
                                equation_str1 = 'y_1 = {0:.5f}x - {1:.5f}'.format(poly1_1[0], abs(poly1_1[n]))
                                equation_str2 = 'y_2 = {0:.5f}x - {1:.5f}'.format(poly1_2[0], abs(poly1_2[n]))
                            equation_str1 += f" (r2_1 = {r2_1_l:.5f})"
                            equation_str2 += f" (r2_2 = {r2_1_r:.5f})"
                            self.printeqnR2.setText(equation_str1 + '\n' + equation_str2)
                        else: # For higher order equations
                            equation_str1 = 'y_1 = {0:.5f}x^{1}'.format(poly1_1[0], n)
                            equation_str2 = 'y_2 = {0:.5f}x^{1}'.format(poly1_2[0], n)
                            for j in range(1, n + 1):
                                coefficient1 = poly1_1[j]
                                power1 = n-j
                                if coefficient1 < 0:
                                    sign = "-"
                                else:
                                    sign = "+"
                                if power1 == 0:
                                    # Appending the term to the equation string
                                    equation_str1 += f" {sign} {abs(coefficient1):.5f}"
                                elif power1 == 1:
                                    equation_str1 += f" {sign} {abs(coefficient1):.5f}x"
                                else:
                                    equation_str1 += f" {sign} {abs(coefficient1):.5f}x^{power1}"
                            equation_str1 += f" (r2_1 = {r2_1_l:.5f})"
                            for j in range(1, n + 1):
                                coefficient2 = poly1_2[j]
                                power2 = n-j
                                if coefficient2 < 0:
                                    sign = "-"
                                else:
                                    sign = "+"
                                if power2 == 0:
                                    # Appending the term to the equation string
                                    equation_str2 += f" {sign} {abs(coefficient2):.5f}"
                                elif power1 == 1:
                                    equation_str2 += f" {sign} {abs(coefficient2):.5f}x"
                                else:
                                    equation_str2 += f" {sign} {abs(coefficient2):.5f}x^{power2}"
                            equation_str2 += f" (r2_2 = {r2_1_r:.5f})"
                        self.printeqnR2.setText(equation_str1 + '\n' + equation_str2)
                    
                else:
                    print(BC)
                    # Curve fit:
                    poly1 = np.polyfit(x_arr, y_arr, deg = order)
                    xp = np.linspace(min(x_arr), max(x_arr), 50)
                    predict1_l = np.poly1d(poly1)
                    R2 = r2_score(y_arr, predict1_l(x_arr))
                    self.ax.plot(xp, predict1_l(xp), 'r-')
                    n = order
                    if n == 1:
                        if poly1[n] > 0:
                            # Print statement for console
                            equation_str = 'y = {0:.5f}x + {1:.5f}'.format(poly1[0], poly1[n])
                        else:
                            equation_str = 'y = {0:.5f}x - {1:.5f}'.format(poly1[0], abs(poly1[n]))
                        equation_str += f" (r^2 = {R2:.5f})"
                        print(equation_str)
                    else: # For higher order equations
                        equation_str = 'y = {0:.5f}x^{1}'.format(poly1[0], n)
                        for j in range(1, n + 1):
                            coefficient = poly1[j]
                            power = n-j
                            if coefficient < 0:
                                sign = "-"
                            else:
                                sign = "+"
                            if power == 0:
                                # Appending the term to the equation string
                                equation_str += f" {sign} {abs(coefficient):.5f}"
                            elif power == 1:
                                equation_str += f" {sign} {abs(coefficient):.5f}x"
                            else:
                                equation_str += f" {sign} {abs(coefficient):.5f}x^{power}"
                        equation_str += f" (r^2 = {R2:.5f})"
                    self.printeqnR2.setText(equation_str)
                    
            else:
                # Curve fit:
                poly1 = np.polyfit(x_arr, y_arr, deg = order)
                xp = np.linspace(min(x_arr), max(x_arr), 50)
                predict1_l = np.poly1d(poly1)
                R2 = r2_score(y_arr, predict1_l(x_arr))
                self.ax.plot(xp, predict1_l(xp), 'r-')
                n = order
                if n == 1:
                    if poly1[n] > 0:
                        # Print statement for console
                        equation_str = 'y = {0:.5f}x + {1:.5f}'.format(poly1[0], poly1[n])
                    else:
                        equation_str = 'y = {0:.5f}x - {1:.5f}'.format(poly1[0], abs(poly1[n]))
                    equation_str += f" (r^2 = {R2:.5f})"
                    print(equation_str)
                else: # For higher order equations
                    equation_str = 'y = {0:.5f}x^{1}'.format(poly1[0], n)
                    for j in range(1, n + 1):
                        coefficient = poly1[j]
                        power = n-j
                        if coefficient < 0:
                            sign = "-"
                        else:
                            sign = "+"
                        if power == 0:
                            # Appending the term to the equation string
                            equation_str += f" {sign} {abs(coefficient):.5f}"
                        elif power == 1:
                            equation_str += f" {sign} {abs(coefficient):.5f}x"
                        else:
                            equation_str += f" {sign} {abs(coefficient):.5f}x^{power}"
                    equation_str += f" (r^2 = {R2:.5f})"
                self.printeqnR2.setText(equation_str)
            if BC == 0:
                bc = 'Normal'
            elif BC == 1:
                bc = 'Trend'
            else:
                bc = 'Jump'
            title = f"Curve Fit with BC: {bc} and Order = {order}"
            self.ax.set_title(title, fontsize=14)
        
        else:
            
            self.text = custom.replace('^', '**')

            def customeq():
                # Make a list of possible coefficent names without x and e:
                alph = list(map(chr, range(97, 123)))
                alph.remove('e')
                alph.remove('x')
                # Pulls in the text input from global. Need to be careful not to change the text variable anywhere else in the code. 
                global text
                # Split off the 'y = ' from the text:
                eq = self.text.split('= ')[1]
                i = 0
                # For each character in the text, if the character is in the alphabet list (coefficient name list) then assign the next value in the coefficients tuple (*nums) to it and create a cariable.
                for char in eq:
                    if char in alph:
                        i = i + 1
                return i            

            k = customeq()
            in_coeff = [1]*k

            def func(x, *nums):
                # Make a list of possible coefficent names without x and e:
                alph = list(map(chr, range(97, 123)))
                alph.remove('e')
                alph.remove('x')
                # Pulls in the text input from global. Need to be careful not to change the text variable anywhere else in the code. 
                global text
                # Split off the 'y = ' from the text:
                eq = self.text.split('= ')[1]
                i = 0
                # For each character in the text, if the character is in the alphabet list (coefficient name list) then assign the next value in the coefficients tuple (*nums) to it and create a cariable.
                for char in eq:
                    if char in alph:
                        exec(f"{char} = {nums[i]}")
                        i = i + 1
                # Use eval on the text to evaluate the equation and return the result.
                result = eval(eq)
                return result

            x_1 = self.x
            y_1 = self.y
                            
            if BC == 0:
                
                coeff, cov = curve_fit(func, x_1, y_1, p0=in_coeff)
    
                x_list = np.linspace(min(x_1), max(x_1), 50)
                y_list = []
    
                for i in x_list:
                    y = func(i, *coeff)
                    y_list.append(y)
    
                self.ax.plot(x_list, y_list, 'r-')
                
                yr = []
                for i in x_1:
                    yo = func(i, *coeff)
                    yr.append(yo)
                R2 = r2_score(y_1, yr)
                
                prnt_result = 'y = '
    
                alph = list(map(chr, range(97, 123)))
                alph.remove('e')
                alph.remove('x')
                textnew = self.text.replace('**','^')
                # Split off the 'y = ' from the text:
                eq = textnew.split('= ')[1]
                i = 0
                # For each character in the text, if the character is in the alphabet list (coefficient name list) then assign the next value in the coefficients tuple (*nums) to it and create a cariable.
                for char in eq:
                    if char in alph:
                        exec(f"{char} = {coeff[i]}")
                        if coeff[i] > 0:
                            prnt_result = prnt_result + ' + ' + str(round(coeff[i], 5))
                        else:
                            prnt_result = prnt_result + ' - ' + str(round(abs(coeff[i]), 5))
                        i = i + 1
                    elif char in ['*','x','^','1','2','3','4','5','6','7','8','9','0','']:
                        prnt_result = prnt_result + char
                    else:
                        prnt_result = prnt_result
                prnt_result = prnt_result + f" (r^2 = {R2:.5f})"
                print(prnt_result)
                self.printeqnR2.setText(prnt_result)
        
            elif BC == 1:
                
                # Convert them to a pandas dataframe for calculating the moving average - used the pandas rolling.mean() approach because it returns the same number of values as there were at the start (good for plotting the same x values) since the function takes averages over a number of points so some points get "lost" - those points are replaces with NaN points (not a number).
                data = pd.DataFrame({'x': x_arr, 'y': y_arr})
        
                # Apply moving average using average data rolling from pandas - two averages for better smoothing, one with a window of 5 points and one with an average of 8 - bigger window makes it smoother but offsets it more.
                data_smooth = data.rolling(window = 5).mean()
                data_smooth2 = data_smooth.rolling(window = 8).mean()
        
                # Find the maximum point - the boundary condition - in the smoothed data set, and plot it to check.
                max_id = list(data_smooth2[['y']].idxmax())[0]
                print(max_id)
                
                # Split the original data set using the new max_id index:    
                x_1 = self.x[:max_id].copy()
                x_2 = self.x[max_id:].copy()
                y_1 = self.y[:max_id].copy()
                y_2 = self.y[max_id:].copy()
                
                coeff1, cov1 = curve_fit(func, x_1, y_1, p0=in_coeff)
                coeff2, cov2 = curve_fit(func, x_2, y_2, p0=in_coeff)
                
                x_list1 = np.linspace(min(x_1), max(x_1), 50)
                y_list1 = []

                x_list2 = np.linspace(min(x_2), max(x_2), 50)
                y_list2 = []
    
                for i in x_list1:
                    y1 = func(i, *coeff1)
                    y_list1.append(y1)
    
                for j in x_list2:
                    y2 = func(j, *coeff2)
                    y_list2.append(y2)
    
                self.ax.plot(x_list1, y_list1, 'r-')
                self.ax.plot(x_list2, y_list2, 'r-')
    
                yr1 = []
                for i in x_1:
                    yo1 = func(i, *coeff1)
                    yr1.append(yo1)
                R21 = r2_score(y_1, yr1)
                
                yr2 = []
                for i in x_2:
                    yo2 = func(i, *coeff2)
                    yr2.append(yo2)
                R22 = r2_score(y_2, yr2)
    
                prnt_result1 = 'y_1 = '
                prnt_result2 = 'y_2 = '
    
                alph = list(map(chr, range(97, 123)))
                alph.remove('e')
                alph.remove('x')
                textnew = self.text.replace('**','^')
                # Split off the 'y = ' from the text:
                eq = textnew.split('= ')[1]
                i = 0
                # For each character in the text, if the character is in the alphabet list (coefficient name list) then assign the next value in the coefficients tuple (*nums) to it and create a cariable.
                for char in eq:
                    if char in alph:
                        exec(f"{char} = {coeff1[i]}")
                        if coeff1[i] > 0:
                            prnt_result1 = prnt_result1 + ' + ' + str(round(coeff1[i], 5))
                        else:
                            prnt_result1 = prnt_result1 + ' - ' + str(round(abs(coeff1[i]), 5))
                        i = i + 1
                    elif char in ['*','x','^','1','2','3','4','5','6','7','8','9','0','']:
                        prnt_result1 = prnt_result1 + char
                    else:
                        prnt_result1 = prnt_result1
                prnt_result1 = prnt_result1 + f" (r^2 = {R21:.5f})"
                print(prnt_result1)
                i = 0
                for char in eq:
                    if char in alph:
                        exec(f"{char} = {coeff2[i]}")
                        if coeff2[i] > 0:
                            prnt_result2 = prnt_result2 + ' + ' + str(round(coeff2[i], 5))
                        else:
                            prnt_result2 = prnt_result2 + ' - ' + str(round(abs(coeff2[i]), 5))
                        i = i + 1
                    elif char in ['*','x','^','1','2','3','4','5','6','7','8','9','0','']:
                        prnt_result2 = prnt_result2 + char
                    else:
                        prnt_result2 = prnt_result2
                prnt_result2 = prnt_result2 + f" (r^2 = {R22:.5f})"
                print(prnt_result2)
                self.printeqnR2.setText(prnt_result1 + '\n' + prnt_result2)
            
            else:
                # Calculate the moving averages of every 4 points
                window_size = 4
                moving_averages = np.convolve(y_arr, np.ones(window_size)/window_size, mode='valid')
                
                # Calculate errors as the differences between original data and moving averages
                errors = y_arr[window_size-1:] - moving_averages
                
                TV = self.TValue.toPlainText()
                if not TV == '':  # Check if TV input exists
                    threshold = float(TV)
                else:
                    # Set threshold dynamically based on the standard deviation of errors
                    threshold = 2*(np.std(errors))
                print(threshold)
                
                # Find indices where the error exceeds the threshold
                jump_indices = np.where(np.abs(errors) > threshold)[0] + window_size - 1
            
                # Print the detected jump indices
                print("Detected jump indices:", jump_indices)
                
                if len(jump_indices) > 0 and len(jump_indices) < 3:
                
                    max_id = jump_indices[0]
                    # Split the original data set using the new max_id index:    
                    x_1 = self.x[:max_id].copy()
                    x_2 = self.x[max_id:].copy()
            
                    y_1 = self.y[:max_id].copy()
                    y_2 = self.y[max_id:].copy()
                
                    coeff1, cov1 = curve_fit(func, x_1, y_1, p0=in_coeff)
                    coeff2, cov2 = curve_fit(func, x_2, y_2, p0=in_coeff)
                    
                    x_list1 = np.linspace(min(x_1), max(x_1), 50)
                    y_list1 = []
    
                    x_list2 = np.linspace(min(x_2), max(x_2), 50)
                    y_list2 = []
        
                    for i in x_list1:
                        y1 = func(i, *coeff1)
                        y_list1.append(y1)
        
                    for j in x_list2:
                        y2 = func(j, *coeff2)
                        y_list2.append(y2)
        
                    self.ax.plot(x_list1, y_list1, 'r-')
                    self.ax.plot(x_list2, y_list2, 'r-')
        
                    yr1 = []
                    for i in x_1:
                        yo1 = func(i, *coeff1)
                        yr1.append(yo1)
                    R21 = r2_score(y_1, yr1)
                    
                    yr2 = []
                    for i in x_2:
                        yo2 = func(i, *coeff2)
                        yr2.append(yo2)
                    R22 = r2_score(y_2, yr2)
        
                    prnt_result1 = 'y_1 = '
                    prnt_result2 = 'y_2 = '
    
        
                    alph = list(map(chr, range(97, 123)))
                    alph.remove('e')
                    alph.remove('x')
                    textnew = self.text.replace('**','^')
                    # Split off the 'y = ' from the text:
                    eq = textnew.split('= ')[1]
                    i = 0
                    # For each character in the text, if the character is in the alphabet list (coefficient name list) then assign the next value in the coefficients tuple (*nums) to it and create a cariable.
                    for char in eq:
                        if char in alph:
                            exec(f"{char} = {coeff1[i]}")
                            if coeff1[i] > 0:
                                prnt_result1 = prnt_result1 + ' + ' + str(round(coeff1[i], 5))
                            else:
                                prnt_result1 = prnt_result1 + ' - ' + str(round(abs(coeff1[i]), 5))
                            i = i + 1
                        elif char in ['*','^','x','1','2','3','4','5','6','7','8','9','0','']:
                            prnt_result1 = prnt_result1 + char
                        else:
                            prnt_result1 = prnt_result1
                    prnt_result1 = prnt_result1 + f" (r^2 = {R21:.5f})"
                    print(prnt_result1)
                    i = 0
                    for char in eq:
                        if char in alph:
                            exec(f"{char} = {coeff2[i]}")
                            if coeff2[i] > 0:
                                prnt_result2 = prnt_result2 + ' + ' + str(round(coeff2[i], 5))
                            else:
                                prnt_result2 = prnt_result2 + ' - ' + str(round(abs(coeff2[i]), 5))
                            i = i + 1
                        elif char in ['*','x','1','2','3','4','5','6','7','8','9','0','']:
                            prnt_result2 = prnt_result2 + char
                        else:
                            prnt_result2 = prnt_result2
                    prnt_result2 = prnt_result2 + f" (r^2 = {R22:.5f})"
                    print(prnt_result2)
                    self.printeqnR2.setText(prnt_result1 + '\n' + prnt_result2)
                
                elif len(jump_indices) > 2:
                    
                    max_id = jump_indices[0]
                    next_id = jump_indices[2]
                    # Split the original data set using the new max_id index:    
                    x_1 = self.x[:max_id].copy()
                    x_2 = self.x[max_id:next_id].copy()
                    x_3 = self.x[next_id:].copy()
            
                    y_1 = self.y[:max_id].copy()
                    y_2 = self.y[max_id:next_id].copy()
                    y_3 = self.y[next_id:].copy()
                    
                    coeff1, cov1 = curve_fit(func, x_1, y_1, p0=in_coeff)
                    coeff2, cov2 = curve_fit(func, x_2, y_2, p0=in_coeff)
                    coeff3, cov3 = curve_fit(func, x_3, y_3, p0=in_coeff)
                    
                    x_list1 = np.linspace(min(x_1), max(x_1), 50)
                    y_list1 = []
    
                    x_list2 = np.linspace(min(x_2), max(x_2), 50)
                    y_list2 = []
                    
                    x_list3 = np.linspace(min(x_3), max(x_3), 50)
                    y_list3 = []
        
                    for i in x_list1:
                        y1 = func(i, *coeff1)
                        y_list1.append(y1)
        
                    for j in x_list2:
                        y2 = func(j, *coeff2)
                        y_list2.append(y2)
                        
                    for p in x_list3:
                        y3 = func(p, *coeff3)
                        y_list3.append(y3)
        
                    self.ax.plot(x_list1, y_list1, 'r-')
                    self.ax.plot(x_list2, y_list2, 'r-')
                    self.ax.plot(x_list3, y_list3, 'r-')
        
                    yr1 = []
                    for i in x_1:
                        yo1 = func(i, *coeff1)
                        yr1.append(yo1)
                    R21 = r2_score(y_1, yr1)
                    
                    yr2 = []
                    for i in x_2:
                        yo2 = func(i, *coeff2)
                        yr2.append(yo2)
                    R22 = r2_score(y_2, yr2)
        
                    yr3 = []
                    for i in x_3:
                        yo3 = func(i, *coeff3)
                        yr3.append(yo3)
                    R23 = r2_score(y_3, yr3)       
        
                    prnt_result1 = 'y_1 = '
                    prnt_result2 = 'y_2 = '
                    prnt_result3 = 'y_3 = '
    
        
                    alph = list(map(chr, range(97, 123)))
                    alph.remove('e')
                    alph.remove('x')
                    textnew = self.text.replace('**','^')
                    # Split off the 'y = ' from the text:
                    eq = textnew.split('= ')[1]
                    i = 0
                    # For each character in the text, if the character is in the alphabet list (coefficient name list) then assign the next value in the coefficients tuple (*nums) to it and create a cariable.
                    for char in eq:
                        if char in alph:
                            exec(f"{char} = {coeff1[i]}")
                            if coeff1[i] > 0:
                                prnt_result1 = prnt_result1 + ' + ' + str(round(coeff1[i], 5))
                            else:
                                prnt_result1 = prnt_result1 + ' - ' + str(round(abs(coeff1[i]), 5))
                            i = i + 1
                        elif char in ['*','^','x','1','2','3','4','5','6','7','8','9','0','']:
                            prnt_result1 = prnt_result1 + char
                        else:
                            prnt_result1 = prnt_result1
                    prnt_result1 = prnt_result1 + f" (r^2 = {R21:.5f})"
                    print(prnt_result1)
                    i = 0
                    for char in eq:
                        if char in alph:
                            exec(f"{char} = {coeff2[i]}")
                            if coeff2[i] > 0:
                                prnt_result2 = prnt_result2 + ' + ' + str(round(coeff2[i], 5))
                            else:
                                prnt_result2 = prnt_result2 + ' - ' + str(round(abs(coeff2[i]), 5))
                            i = i + 1
                        elif char in ['*','x','1','2','3','4','5','6','7','8','9','0','']:
                            prnt_result2 = prnt_result2 + char
                        else:
                            prnt_result2 = prnt_result2
                    prnt_result2 = prnt_result2 + f" (r^2 = {R22:.5f})"
                    print(prnt_result2)
                    i = 0
                    for char in eq:
                        if char in alph:
                            exec(f"{char} = {coeff3[i]}")
                            if coeff3[i] > 0:
                                prnt_result3 = prnt_result3 + ' + ' + str(round(coeff3[i], 5))
                            else:
                                prnt_result3 = prnt_result3 + ' - ' + str(round(abs(coeff3[i]), 5))
                            i = i + 1
                        elif char in ['*','x','1','2','3','4','5','6','7','8','9','0','']:
                            prnt_result3 = prnt_result3 + char
                        else:
                            prnt_result3 = prnt_result3
                    prnt_result3 = prnt_result3 + f" (r^2 = {R23:.5f})"
                    print(prnt_result3)
                    self.printeqnR2.setText(prnt_result1 + '\n' + prnt_result2 + '\n' + prnt_result3)
                    
            if BC == 0:
                bc = 'Normal'
            elif BC == 1:
                bc = 'Trend'
            else:
                bc = 'Jump'
            title = f"Curve Fit with BC = {bc}, Input: {custom}"
            self.ax.set_title(title, fontsize=14)
              
        self.ax.set_xlabel('x', fontsize=14)
        self.ax.set_ylabel('y', fontsize=14)
        self.canvas.draw()
    
    def clear(self):
        self.ShowCB.setChecked(False)
        self.Points.clear()
        self.OrderSpinBox.clear()
        self.BCspinBox.clear()
        self.textequation.clear()
        self.xyinput.clear()
        self.TValue.clear()
        self.printeqnR2.clear()
        for item in self.scene.items():
            self.scene.removeItem(item)


    def closeprog(self):
        self.close()
                  
    def savepng(self):
        # Set the working directory to the directory containing the script
        script_path = os.path.abspath(__file__)
        script_directory = os.path.dirname(script_path)
        os.chdir(script_directory)
        # Save the figure to the working directory
        save_path = os.path.join(script_directory, 'trial.png')
        self.fig.savefig(save_path)
                
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())