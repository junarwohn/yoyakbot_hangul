from pytesseract import *
import cv2
import os
import re
import numpy as np
import difflib
from difflib import SequenceMatcher

file_list = os.listdir("extract/")
file_list.sort()
for file_name in file_list:
    original_img = cv2.imread("extract/" + file_name)
    cv2.imshow("original", original_img)
    ok = cv2.waitKey(0)
    if ok != 13:
        continue
cv2.destroyAllWindows()
