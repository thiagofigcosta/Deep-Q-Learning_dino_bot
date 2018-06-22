#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pyautogui

pyautogui.moveTo(100, 100, duration=0.25) #move mouse to
pyautogui.moveRel(100, 0, duration=0.25) #sum to mouse position
print (pyautogui.position()) #get mouse position

pyautogui.mouseDown() #push mouse button
pyautogui.mouseUp() #release mouse button

pyautogui.click() #click at current position
pyautogui.doubleClick() #double click
pyautogui.click(100, 150, button='left') #'left', 'middle', 'right' or pyautogui.middleClick() or pyautogui.rightClick() 

pyautogui.dragTo(100, 100, duration=0.25) #click and move to
pyautogui.dragRel(100, 100, duration=0.25) #click and sum to position

pyautogui.scroll(200) #scroll mouse