#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pyautogui

im = pyautogui.screenshot()

print (im.getpixel((0, 0)))

pyautogui.pixelMatchesColor(50, 200, (130, 135, 144))

pyautogui.locateOnScreen('file.png') #locate this image on the screen
list(pyautogui.locateAllOnScreen('file.png')) #locate this image on the screen
pyautogui.center(pyautogui.locateOnScreen('file.png')) #center of the iamge