Team Pynapples - Lightning Talk Presentation

Package: PyQt5

Follow the link to access the Instructions page:
https://docs.google.com/document/d/1sYwtFDPpKuq89KZXBkEbIZP7RiYTCpaZ6lGOF3qcyfg/edit?usp=sharing

---------------------------------------------------------------------------------------------------------------------

Final Project:
Download the code 'finalprojectCT.py' and the ui design file 'CFA.ui' to run the code successfully.

The excel files 'book1 - book4' are the files to upload in the GUI for fitting.

You can add new files or input points manually as well in the specified format.


HOW TO:


- INSTALL PYQT5 & QT DESIGNER


Anaconda on Windows
PyQt is already integrated with Spyder/Anaconda. If you’re running the code with spyder on windows, installation is not necessary. For operating systems, PyQt has to be installed before running the code which might involve restarting the system as well.


Installation
The simplest way to install PyQt5 on Linux/windows or macOS is to use Python's pip packaging tool, just as for other packages. For Python3 installations this is usually called pip3.

After installing, you should be able to run python and import PyQt5. Note that if you want to access Qt Designer or Qt Creator you will need to download this from https://qt.io/download [the Qt downloads site]. You do not need Qt Designer to run our demo. It’s only necessary when creating new GUI designs.


- USE THE GUI


Download demo: 
From the Github repository link above, download the code ‘finalprojectCT.py’ and the GUI design file ‘CFA.ui’ into the same folder on your PC. Try curve fits for any random data sets or use the available excel files ‘Book 1, Book 2,…’ by downloading them to the same folder.


Available options:-

Input points: Two methods, either through a .xlsx file upload with the x and y points in the first 2 columns, or manually entering points in the text box in x, y format.

Order of fit: Choose from 1-10. It is set by default to order ‘1’ (linear). Check to make sure these boxes are always populated with the desired values. When a custom equation is given, the input order box value is neglected.

Type of Boundary Condition: Choose from None, Trend and Jump as per requirement. Select ‘0’ (None) for a simple fit without considering jumps or trends. Select ‘1’ (Trend) to detect a change in the trend of the data (increasing to decreasing, or decreasing to increasing) and select ‘2’ (Jump) to detect gaps in the data based on the threshold value.

Threshold value for BC Jump: TV can be manually input to a value based on desired fit type or in to neglect unintended gaps in the curve. The code will then only detect gaps which are more than the input threshold value. If the TV field is left empty, jumps are detected based on the standard deviation of errors in the data by default (only if BC = 2).

Custom equation text input: Manually enter the equation type (linear or higher order) in the text box. For example, a 3rd order equation would be ‘y = a*x^3 + b*x + d’.

Save Plot: Clicking the ‘Save’ button saves the plot to the code working directory location.

Display Confidence Bounds (CB): 95% confidence intervals and prediction intervals are plotted to the curve based on the fit type. This can be enabled by checking the ‘C B’ check box.

Fit equation: The curve fit equations are displayed in the ‘equation’ box above the plot along with their r2 value (coefficient of determination). When r2= 1, no confidence bounds are plotted as it’s a perfect fit.

Clear: Erases all the inputs and displays of the GUI to start over.


CHOOSING THE RIGHT FIT

Judge the final plot based on your requirements and through the following:

Coefficient of determination: R2 value (the closer it is to 1, the better the fit)

Order of fit (higher order curves fit better but their equations are computationally complex for future predictions)

Boundary Conditions (Not all curves work for all kinds of data, if the plot is empty or not in the desired manner, change the BC and order values to get the best one)

Play around with all the options to find the required fits and choose what’s best for you.


LIMITATIONS:


Data Type:

For the Trend BC: Only one Increasing - decreasing /(or) Decreasing - increasing trend can be identified with BC = 1. For example, for a sine wave with multiple ups and downs, multiple curves aren’t generated. The code is limited to a simple ‘U’ or ‘inverted U’ type data sets.

For the Jump BC: The code is constructed to identify only up to 2 gaps/breaks in the data. If the data set has more than 2 large jumps, only the first two curves would fit correctly and the last one would include the rest of the data in a single curve.

Future improvements could certainly include making the code more flexible to ‘n’ number of gaps and trends but it was out of scope for our current project.

Certain combinations of input might not generate plots depending upon fit types and compatibility. The user has to use their best judgment and change/choose the boundary condition and order values accordingly.

Custom equation text input:

The custom equation can be input in virtually any form as long as the characters used are python recognized, with one exception: the ^ symbol has been built into the program instead of **.

Spacing does not matter for the most part, however, a space must be used between the = and the rest of the expression due to how the code was written. 

This can be changed by editing the equation function to use ‘=’ instead of ‘= ‘ for the split and by utilizing strip() on the text after splitting.

The custom text input equation code is currently unable to take an input that has an ‘e’ in the equation. However, since the numpy package is utilized in the program code, it can recognize np.e as e. Because of this, np.e should be utilized in the text input instead.

To simplify this, an additional line of code can be added to the custom equation text input code:

Within the text input string, replace e with np.e.

A similar piece of code is already currently implemented to allow the user to type in the ^ symbol rather than using **.


Follow the link to access the Instructions/Report page:

https://docs.google.com/document/d/1fQR8wsq5lEA5E-BxqpOA4gX_Fjq1FBxM/edit

