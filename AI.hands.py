#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyautogui

#keyboard and mouse controller 

class Hands(object):
	def __init__(self, arg):
		self.arg = arg
		
	def moveMouse(self,x,y,duration=0,rel=False):
		if rel:
			pyautogui.moveRel(x, y, duration=duration)
		else:	
			pyautogui.moveTo(x, y, duration=duration)

	def clickMouse(self,left=False,right=False,middle=False,double=False):
		if left==right==middle==False:
			left=True
		if right:
			if double:
				pyautogui.doubleClick(button='right') 
			else:
				pyautogui.click(button='right') 
		elif middle:
			if double:
				pyautogui.doubleClick(button='middle') 
			else:
				pyautogui.click(button='middle') 
		else:
			if double:
				pyautogui.doubleClick(button='left') 
			else:
				pyautogui.click(button='left') 

	def putMouseOnOrigin(self,screen):
		o=screen.getOrigin()
		self.moveMouse(o.x,o.y)

	def focousScreen(self,screen):
		s=screen.getSize()
		o=screen.getOrigin()
		self.moveMouse(o.x+s.x/2, o.y+s.y/2)
		self.clickMouse()

	def pressKeyboard(self,key):
		pyautogui.press(key)