#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pyautogui

pyautogui.typewrite('test') #writes this
pyautogui.typewrite('test', 0.25) #writes this with delay
pyautogui.typewrite(['a', 'enter', 'left', 'left', 'shiftleft', 'altleft', 'tab', 'ctrlright', 'backspace', 'home', 'volumemute', 'insert', 'winleft', 'command'])

pyautogui.press('4')

pyautogui.keyDown('ctrl')
pyautogui.keyDown('c')
pyautogui.keyUp('c')
pyautogui.keyUp('ctrl')

pyautogui.hotkey('ctrl', 'c')