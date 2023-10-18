# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:37:24 2023

@author: jahna
"""

import sys
from PyQt5 import QtWidgets, uic
import random

qtcreator_file  = "bmilayout5.ui" # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.CalcButton.clicked.connect(self.CalculateBMI)
        self.TranslateButton.clicked.connect(self.Translate)
        self.BMI = 0
        self.GrootButton.clicked.connect(self.Grootsays)
        self.ClearButton.clicked.connect(self.clear)
        self.ExitButton.clicked.connect(self.closeprog)
        
    def CalculateBMI(self):
        H = float(self.height.toPlainText())
        W = float(self.weight.toPlainText())
        BMI = round(W/(H**2),1)
        output_string = str(BMI)
        self.output.setText(output_string)
        return BMI

    def Grootsays(self):
        self.comment.setText('I am Groot!')
        
    def Translate(self):
        compliments = ['You look like you can dance!','You look amazing today!',
                       'Would you like a tour of the galaxy?!',
                       'I love that shirt on you!','Im so proud of you!','You are Gorgeous!']
        n = random.randint(0,5)
        string = str(compliments[n])
        self.trancomm.setText(string)

    def clear(self):
        self.height.clear()
        self.weight.clear()
        self.output.clear()
        self.comment.clear()
        self.trancomm.clear()

    def closeprog(self):
        self.close()
                    
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())