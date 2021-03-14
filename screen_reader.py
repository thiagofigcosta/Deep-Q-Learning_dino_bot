#!/bin/python

import numpy as np
import pyautogui,cv2,time

def pointsToRectangle(x0,y0,x1,y1):
    w=int(x1-x0)
    h=int(y1-y0)
    rec={'x0':int(x0),'x1':int(x1),'y0':int(y0),'y1':int(y1),'w':int(w),'h':int(h)}
    return rec

def setup():
    def draw_rect(event,x,y,flags,param):
        nonlocal x0,y0,x1,y1,drawing,screen_bkp,screen,selected_square,thickness,color
        if event==cv2.EVENT_LBUTTONDOWN:
            drawing=True
            x0,y0=x,y
            # screen_bkp=screen.copy()
        elif event==cv2.EVENT_MOUSEMOVE:
            if drawing==True:
                if screen_bkp is not None:
                    screen=screen_bkp.copy()
                cv2.rectangle(screen,(x0,y0),(x,y),color,thickness)
        elif event==cv2.EVENT_LBUTTONUP:
            if drawing==True:
                drawing=False
                selected_square=True
                x1,y1=x,y
                cv2.rectangle(screen,(x0,y0),(x,y),color,thickness)
    thickness=2
    color=(0,255,0)
    x0,y0,x1,y1=-1,-1,-1,-1
    drawing=False
    selected_square=False
    delay=3
    print('Taking screenshot after {} seconds'.format(delay))
    time.sleep(delay)
    screen=cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
    screen_bkp=screen.copy()
    cv2.namedWindow('screen', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while not selected_square:
        cv2.setMouseCallback('screen',draw_rect)
        cv2.imshow('screen',screen)
        cv2.waitKey(1)
    print('Found game screen ({}, {}), ({}, {})'.format(x0,y0,x1,y1))
    screen=screen_bkp.copy()
    cv2.rectangle(screen,(x0,y0),(x1,y1),(0,255,0),2)
    cv2.imshow('screen',screen)
    time.sleep(1)
    cv2.destroyWindow('screen')
    cv2.waitKey(1)
    if x1<x0:
        tmp=x0
        x0=x1
        x1=tmp
    if y1<y0:
        tmp=y0
        y0=y1
        y1=tmp
    return pointsToRectangle(x0,y0,x1,y1)


def captureScreen(area_rec):
    screen=cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
    return screen[area_rec['y0']:area_rec['y1'],area_rec['x0']:area_rec['x1']]