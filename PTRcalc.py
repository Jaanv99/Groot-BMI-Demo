# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 12:19:08 2023

@author: jahna
"""

import sys
from PyQt5 import QtWidgets, uic

qtcreator_file  = "calctemp.ui" # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.CalculateTax)
        
    def CalculateTax(self):
        price = int(self.principal.toPlainText())
        tax = (self.interest.value())
        total_price = price  + ((tax / 100) * price)
        total_price_string = "The total price with tax is: " + str(total_price)
        self.output.setText(total_price_string)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())