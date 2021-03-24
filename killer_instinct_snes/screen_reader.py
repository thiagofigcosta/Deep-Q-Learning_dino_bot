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
    print('Taking screenshot in {} seconds'.format(delay))
    time.sleep(delay)
    screen=cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
    screen_bkp=screen.copy()
    cv2.namedWindow('screen', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while not selected_square:
        cv2.setMouseCallback('screen',draw_rect)
        cv2.imshow('screen',screen)
        cv2.waitKey(1)
    if x1<x0:
        tmp=x0
        x0=x1
        x1=tmp
    if y1<y0:
        tmp=y0
        y0=y1
        y1=tmp
    print('Found game screen ({}, {}), ({}, {})'.format(x0,y0,x1,y1))
    screen=screen_bkp.copy()
    cv2.rectangle(screen,(x0,y0),(x1,y1),(0,255,0),2)
    cv2.imshow('screen',screen)
    time.sleep(1)
    cv2.destroyWindow('screen')
    cv2.waitKey(1)
    return pointsToRectangle(x0,y0,x1,y1)


def captureScreen(area_rec,grey=False):
    screen=np.array(pyautogui.screenshot())
    screen=screen[area_rec['y0']:area_rec['y1'],area_rec['x0']:area_rec['x1']]
    if grey:
        screen=cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    else:
        screen=cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
    return screen 


def isRelativeltyEqual(a,b,confidance):
    confidance=1-confidance
    return a*(1-confidance)<=b and a*(1+confidance)>=b

def matchMaskedSprite(image,sprite,confidance,mask=None): # too slow
    if len(image.shape)==2:
        image=np.reshape(image,image.shape+(1,))
    if len(sprite.shape)==2:
        sprite=np.reshape(sprite,sprite.shape+(1,))
    image_h,image_w,image_c=image.shape
    sprite_h,sprite_w,sprite_c=sprite.shape
    if mask is not None:
        if len(mask.shape)==2:
            mask=np.reshape(mask,mask.shape+(1,))
        mask_h,mask_w,mask_c=mask.shape
        if mask_h!=sprite_h or mask_w!=sprite_w:
            raise 'Wrong mask size'
    if image_c!=sprite_c:
        raise 'Wrong channels size'
    match_points=[]
    # output=np.zeros((image_h,image_w))
    for i in range(image_h-sprite_h):
        for j in range(image_w-sprite_w):
            have_match=True
            for c in range(sprite_h):
                for t in range(sprite_w):
                    if mask is None or mask[c][t][0]==255:
                        for k in range(image_c):
                            if not isRelativeltyEqual(image[i+c][j+t][k],sprite[c][t][k],confidance):
                                have_match=False
                                break
                        if not have_match:
                            break
                    if not have_match:
                        break
                if not have_match:
                    break
            if have_match:
                print('we have a match')
                match_points.append((i,j))
            # output[i][j]=0
    # return output
    return match_points