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

def getContoursOfImage(img,show_edges=False,name='Image',detect_edges=False,display_size=None):
    img_edges=auto_canny(img) # edge detection
    if show_edges:
        to_show=img_edges.copy()
        if display_size !=None:
            to_show=resizeImg(to_show,display_size)
        cv2.imshow('{} - edge'.format(name), to_show)
    cnts,hierarchy=cv2.findContours(img_edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    return cnts

def checkIfRectanglesIntersects(x_0,y_0,w_0,h_0,x_1,y_1,w_1,h_1):
    x0_0=x_0
    x1_0=x_0+w_0
    y0_0=y_0
    y1_0=y_0+h_0
    rec_0={'x0':x0_0,'x1':x1_0,'y0':y0_0,'y1':y1_0,'w':w_0,'h':h_0}
    x0_1=x_1
    x1_1=x_1+w_1
    y0_1=y_1
    y1_1=y_1+h_1
    rec_1={'x0':x0_1,'x1':x1_1,'y0':y0_1,'y1':y1_1,'w':w_1,'h':h_1}
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
        top_rec=rec_1
        low_rec=rec_0
    else:
        top_rec=rec_0
        low_rec=rec_1
    # if (left_rec['x1'] <= right_rec['x0']) or (low_rec['y1'] <= top_rec['y0']):
    #     return 0 # does not overlap
    # elif (left_rec['x1'] <= right_rec['x1']) or (low_rec['y1'] <= top_rec['y1']): 
    #     if w_0 > w_1 or h_0 > h_1:
    #         return 2 # full overlap, contains - 0 is bigger
    #     else:
    #         return 3 # full overlap, contains - 1 is bigger
    # else: 
    #     return 1 # intersect
    if (rec_1['x1'] > rec_0['x0'] and rec_1['x1'] < rec_0['x1']) or (rec_1['x0'] > rec_0['x0'] and rec_1['x0'] < rec_0['x1']):
        x_match = True
    else:
        x_match = False
    if (rec_1['y1'] > rec_0['y0'] and rec_1['y1'] < rec_0['y1']) or (rec_1['y0'] > rec_0['y0'] and rec_1['y0'] < rec_0['y1']):
        y_match = True
    else:
        y_match = False
    if x_match and y_match:
        return True
    else:
        return False

def getBiggestRec(x_0,y_0,w_0,h_0,x_1,y_1,w_1,h_1):
    x0_0=x_0
    x1_0=x_0+w_0
    y0_0=y_0
    y1_0=y_0+h_0
    rec_0={'x0':x0_0,'x1':x1_0,'y0':y0_0,'y1':y1_0,'w':w_0,'h':h_0}
    x0_1=x_1
    x1_1=x_1+w_1
    y0_1=y_1
    y1_1=y_1+h_1
    rec_1={'x0':x0_1,'x1':x1_1,'y0':y0_1,'y1':y1_1,'w':w_1,'h':h_1}
    small_x=None
    big_x=None
    if rec_0['x0'] < rec_1['x0']:
        small_x=rec_0['x0']
    else:
        small_x=rec_1['x0']
    if rec_0['x1'] > rec_1['x1']:
        big_x=rec_0['x1']
    else:
        big_x=rec_1['x1']
    small_y=None
    big_y=None
    if rec_0['y0'] < rec_1['y0']:
        small_y=rec_0['y0']
    else:
        small_y=rec_1['y0']
    if rec_0['y1'] > rec_1['y1']:
        big_y=rec_0['y1']
    else:
        big_y=rec_1['y1']
    return small_x,small_y,big_x-small_x,big_y-small_y

def simplifyOverlappingContours(cnts):
    final_boundaries=[]
    already_simplified=set()
    for i,cnt_a in enumerate(cnts):
        if i not in already_simplified:
            x_0,y_0,w_0,h_0=cv2.boundingRect(cnt_a)
            for j,cnt_b in enumerate(cnts):
                if j>i and j not in already_simplified:
                    x_1,y_1,w_1,h_1=cv2.boundingRect(cnt_b)
                    case=checkIfRectanglesIntersects(x_0,y_0,w_0,h_0,x_1,y_1,w_1,h_1)
                    if case:
                        already_simplified.add(j) # ignores j
                        x_0,y_0,w_0,h_0=getBiggestRec(x_0,y_0,w_0,h_0,x_1,y_1,w_1,h_1)
                    # if case==0:
                    #     pass # ignore
                    # elif case==1:
                    #     already_simplified.add(j) # ignores j
                    #     x_0,y_0,w_0,h_0=getBiggestRec(x_0,y_0,w_0,h_0,x_1,y_1,w_1,h_1)
                    # elif case==2:
                    #     already_simplified.add(j) # ignores j
                    # elif case==3:
                    #     already_simplified.add(j) # ignores j
                    #     x_0=x_1
                    #     y_0=y_1
                    #     w_0=w_1
                    #     h_0=h_1
            already_simplified.add(i)
            final_boundaries.append((x_0,y_0,w_0,h_0))
    return final_boundaries

def loadSpriteSheet(path,threshold=1,display=False):
    display_size=800
    thickness=2
    sprite_name=getFilename(path)
    img_spr_sheet=cv2.imread(path)
    img_spr_sheet=clearBackground(img_spr_sheet,[255,0,255])
    img_spr_sheet_bin=cv2.cvtColor(img_spr_sheet.copy(),cv2.COLOR_BGR2GRAY)
    img_spr_sheet_bin=greyToBinary(img_spr_sheet_bin,threshold)
    img_spr_sheet_bin=morphologicalTransformation(img_spr_sheet_bin,kernel_size=3)
    if display:
        to_show=resizeImg(img_spr_sheet_bin,display_size)
        cv2.imshow('{} - bin'.format(sprite_name), to_show)
    contours=getContoursOfImage(img_spr_sheet_bin,show_edges=display,name=sprite_name,display_size=display_size)
    print('Contours before simplify: {}'.format(len(contours)))
    # color=(0,255,0)
    # for cnt_bnd in contours:
    #     x,y,w,h=cv2.boundingRect(cnt_bnd)
    #     img_single_sprite=img_spr_sheet[y:y+h][x:x+w]
    #     cv2.rectangle(img_spr_sheet,(x,y),(x+w,y+h),color, thickness)
    contours_boundaries=simplifyOverlappingContours(contours)
    print('Contours after simplify: {}'.format(len(contours_boundaries)))
    color=(255,0,0)
    for cnt_bnd in contours_boundaries:
        x,y,w,h=cnt_bnd
        img_single_sprite=img_spr_sheet[y:y+h][x:x+w]
        cv2.rectangle(img_spr_sheet,(x,y),(x+w,y+h),color, thickness)
    if display:
        to_show=resizeImg(img_spr_sheet,display_size)
        cv2.imshow('{} - raw with contours'.format(sprite_name), to_show)
    if display:
        cv2.waitKey()

def main(argv):
    path='games/sprites/killer_instinct/Cinder.png'
    loadSpriteSheet(path,display=True)

if __name__=='__main__':
    main(sys.argv[1:])