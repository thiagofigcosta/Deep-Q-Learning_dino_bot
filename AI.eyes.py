#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

#CV, identifies elements on the screen 

class AI_eyes(object):
	def __init__(self, arg):
		self.arg = arg

	def captureScreen(self,screen): #get print using boundaries are faster
		if screen.fullscreen:
			return pyautogui.screenshot()
		else:
			s=screen.getSize()
			o=screen.getOrigin()
			return pyautogui.screenshot(region=(origin.x,origin.y,size.x,size.y))