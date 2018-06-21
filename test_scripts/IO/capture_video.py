#!/usr/bin/env python
# -*- coding: utf-8 -*-
import platform

if platform.system()=='Linux':
	import pyscreenshot as ImageGrab
else:
	from PIL import ImageGrab
	
import numpy as np
import cv2

img = ImageGrab.grab(bbox=(100,10,400,780)) #bbox specifies specific region (bbox= x,y,width,height *starts top-left)
img_np = np.array(img) #this is the array obtained from conversion
frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
cv2.imshow("test", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()