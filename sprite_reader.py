#!/bin/python

import cv2,sys,time,math,os
import random as rd
import numpy as np

def greyToBinaryInline(img_grey,img_bin,threshold,copy=True):
    if copy:
        img_bin=img_grey.copy()
    for i in range(img_grey.shape[1]):
        for j in range(img_grey.shape[0]):
            img_bin[j][i]=255 if img_grey[j][i]>(255-threshold) else 0

def greyToBinary(img_grey,threshold):
    img_bin=img_grey.copy()
    greyToBinaryInline(img_grey,img_bin,threshold,copy=False)
    return img_bin

def clearBackground(img,color_to_find,color_to_replace=(255,255,255)):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            equal=True
            for k in range(len(color_to_find)):
                if img[i][j][k]!=color_to_find[k]:
                    equal=False
                    break
            if equal:
                img[i][j]=color_to_replace
    return img

def getFilename(path,include_ext=False):
    filename=os.path.basename(path)
    if not include_ext:
        filename=filename.rsplit('.',1)[0]
    return filename

def morphologicalTransformation(img_bin,kernel_size):
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size,kernel_size))
    close=cv2.morphologyEx(img_bin,cv2.MORPH_CLOSE,kernel,iterations=3)
    dilate=cv2.dilate(close,kernel,iterations=1)
    return img_bin

def auto_canny(image,sigma=0.33):
    # https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
	# compute the median of the single channel pixel intensities
	v=np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower=int(max(0,(1.0-sigma)*v))
	upper=int(min(255,(1.0+sigma)*v))
	edged=cv2.Canny(image,lower,upper)
	# return the edged image
	return edged

def resizeImg(img,width):
    return cv2.resize(img,(width,int(width*img.shape[0]/img.shape[1])))

def resizeImgH(img,height):
    return cv2.resize(img,(int(height*img.shape[1]/img.shape[0]),height))

def getContoursOfImage(img,show_edges=False,name='Image',detect_edges=False,display_size=None):
    img_edges=auto_canny(img) # edge detection
    if show_edges:
        to_show=img_edges.copy()
        if display_size !=None:
            to_show=resizeImgH(to_show,display_size)
        cv2.imshow('{} - edge'.format(name), to_show)
    cnts,hierarchy=cv2.findContours(img_edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    return cnts

def checkIfRectanglesIntersects(rec_a,rec_b,offset=0):
    rec_0=rec_a.copy()
    rec_1=rec_b.copy()
    if (offset>0):
        offset=int(offset/2)
        rec_0['x0']-=offset
        rec_0['y0']-=offset
        rec_0['x1']+=offset
        rec_0['y1']+=offset
        rec_1['x0']-=offset
        rec_1['y0']-=offset
        rec_1['x1']+=offset
        rec_1['y1']+=offset
    left_rec=None
    right_rec=None
    if rec_0['x0'] < rec_1['x0']:
        left_rec=rec_0
        right_rec=rec_1
    else:
        left_rec=rec_1
        right_rec=rec_0
    top_rec=None
    low_rec=None
    if rec_0['y0'] < rec_1['y0']:
        top_rec=rec_0
        low_rec=rec_1
    else:
        top_rec=rec_1
        low_rec=rec_0
    if (left_rec['x1'] < right_rec['x0']) or (low_rec['y0'] > top_rec['y1']):
        return 0 # does not overlap
    elif (left_rec['x1'] > right_rec['x1']) or (low_rec['y1'] < top_rec['y1']): 
        if rec_0['w'] >= rec_1['w'] or rec_0['h'] >= rec_1['h']:
            return 2 # full overlap, contains - 0 is bigger
        else:
            return 3 # full overlap, contains - 1 is bigger
    else: 
        return 1 # intersect

def getEquivalentRectangles(rec_0,rec_1):
    equivalent=[]
    collision=checkIfRectanglesIntersects(rec_0,rec_1,offset=4)
    if collision==0:
        equivalent.append(rec_0)
        equivalent.append(rec_1)
    elif collision==1:
        x_0=None
        x_1=None
        if rec_0['x0'] < rec_1['x0']:
            x_0=rec_0['x0']
        else:
            x_0=rec_1['x0'] 
        if rec_0['x1'] > rec_1['x1']:
            x_1=rec_0['x1']
        else:
            x_1=rec_1['x1'] 
        y_0=None
        y_1=None
        if rec_0['y0'] < rec_1['y0']:
            y_0=rec_0['y0']
        else:
            y_0=rec_1['y0'] 
        if rec_0['y1'] > rec_1['y1']:
            y_1=rec_0['y1']
        else:
            y_1=rec_1['y1']
        equivalent.append(pointAndSizeToRectangle(x_0,y_0,(x_1-x_0),(y_1-y_0)))
    elif collision==2:
        equivalent.append(rec_0)
    elif collision==3:
        equivalent.append(rec_1)
    return equivalent

def pointAndSizeToRectangle(x,y,w,h):
    rec={'x0':int(x),'x1':int(x+w),'y0':int(y),'y1':int(y+h),'w':int(w),'h':int(h)}
    return rec

def contoursToRectangles(cnts):
    recs=[]
    for cnt in cnts:
        x,y,w,h=cv2.boundingRect(cnt)
        rec=pointAndSizeToRectangle(x,y,w,h)
        recs.append(rec)
    return recs


def enlargeRec(rec,offset):
    if (offset>0):
        offset=int(offset/2)
        rec['x0']-=offset
        rec['y0']-=offset
        rec['x1']+=offset
        rec['y1']+=offset
    return rec

def simplifyOverlappingRectangles(recs):
    final_recs=[]
    already_simplified_indexes=set()
    for i,rec_0 in enumerate(recs):
        if i not in already_simplified_indexes:
            go_again=True
            while go_again:
                go_again=False
                for j,rec_1 in enumerate(recs):
                    if j>i and j not in already_simplified_indexes:
                        eq=getEquivalentRectangles(rec_0,rec_1)
                        if len(eq)==1:
                            rec_0=eq[0]
                            already_simplified_indexes.add(j)
                            go_again=True
                            break
            final_recs.append(rec_0)
    return final_recs

def filterSmallRecs(recs,min_area,min_w,min_h):
    final_recs=[]
    for rec in recs:
        if rec['w']*rec['h']>=min_area and rec['w']>=min_w and rec['h']>=min_h:
            final_recs.append(rec)
    return final_recs

def loadSpriteSheet(path,threshold=1,display=False):
    sprites=[]
    display_size=600
    thickness=2
    sprite_name=getFilename(path)
    img_spr_sheet=cv2.imread(path)
    img_spr_sheet=clearBackground(img_spr_sheet,[255,0,255])
    img_spr_sheet_bin=cv2.cvtColor(img_spr_sheet.copy(),cv2.COLOR_BGR2GRAY)
    img_spr_sheet_bin=greyToBinary(img_spr_sheet_bin,threshold)
    img_spr_sheet_bin=morphologicalTransformation(img_spr_sheet_bin,kernel_size=3)
    if display:
        to_show=resizeImgH(img_spr_sheet_bin,display_size)
        cv2.imshow('{} - bin'.format(sprite_name), to_show)
    contours=getContoursOfImage(img_spr_sheet_bin,show_edges=display,name=sprite_name,display_size=display_size)
    recs=contoursToRectangles(contours)
    print('Contours before simplify: {}'.format(len(recs)))
    recs=simplifyOverlappingRectangles(recs)
    recs=filterSmallRecs(recs,250,20,20)
    print('Contours after simplify: {}'.format(len(recs)))
    color=(255,0,0)
    for rec in recs:
        rec=enlargeRec(rec,4)
        x,y,w,h=rec['x0'],rec['y0'],rec['w'],rec['h']
        img_single_sprite=img_spr_sheet[y:y+h,x:x+w].copy()
        sprites.append(img_single_sprite)
        cv2.rectangle(img_spr_sheet,(x,y),(x+w,y+h),color, thickness)
    if display:
        to_show=resizeImgH(img_spr_sheet,display_size)
        cv2.imshow('{} - raw with contours'.format(sprite_name), to_show)
    if display:
        cv2.waitKey()
    return sprites


def loadAllKillerInstinctSpriteSheets():
    sprites={}
    base_path='games/sprites/killer_instinct/'
    extension='.png'
    sprite_sheets=['Cinder','Combo','Eyedol','Fulgore','Glacius','Jago','Orchid','Riptor','Sabrewulf','Spinal','Thunder']
    for sprite_sheet in sprite_sheets:
        path=base_path+sprite_sheet+extension
        sprites[sprite_sheet]=loadSpriteSheet(path)
    return sprites