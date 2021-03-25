#!/bin/python

import cv2,sys,time,math,os,json,codecs,zlib
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

def noisyBackground(img,color_to_find):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            equal=True
            for k in range(len(color_to_find)):
                if img[i][j][k]!=color_to_find[k]:
                    equal=False
                    break
            if equal:
                if bool(rd.getrandbits(1)):
                    img[i][j]=(0,0,0)
                else:
                    img[i][j]=(255,255,255)
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
    elif collision>=1:
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
    elif collision==2: # TODO not working
        equivalent.append(rec_0)
    elif collision==3: # TODO not working
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

def loadSpriteSheet(path,threshold=1,display=False,grey=False):
    sprites=[]
    display_size=600
    thickness=2
    sprite_name=getFilename(path)
    img_spr_sheet_raw=cv2.imread(path) 
    img_spr_sheet=clearBackground(img_spr_sheet_raw.copy(),[255,0,255])
    img_spr_sheet_grey=cv2.cvtColor(img_spr_sheet,cv2.COLOR_BGR2GRAY)
    img_spr_sheet_bin=greyToBinary(img_spr_sheet_grey,threshold)
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
    img_spr_sheet_noisy=noisyBackground(img_spr_sheet_raw,[255,0,255])
    if grey:
        img_spr_sheet_to_cut=cv2.cvtColor(img_spr_sheet_noisy,cv2.COLOR_BGR2GRAY)
    else:
        img_spr_sheet_to_cut=img_spr_sheet_noisy
    if display:
        to_show=resizeImgH(img_spr_sheet_to_cut,display_size)
        cv2.imshow('{} - to cut'.format(sprite_name), to_show)
    color=(255,0,0)
    for rec in recs:
        rec=enlargeRec(rec,4)
        x0,y0,x1,y1=rec['x0'],rec['y0'],rec['x1'],rec['y1']
        img_single_sprite=img_spr_sheet_to_cut[y0:y1,x0:x1].copy()
        sprites.append(img_single_sprite)
        if display:
            cv2.rectangle(img_spr_sheet,(x0,y0),(x1,y1),color, thickness)
    if display:
        to_show=resizeImgH(img_spr_sheet,display_size)
        cv2.imshow('{} - raw with contours'.format(sprite_name), to_show)
    if display:
        cv2.waitKey()
    return sprites


def loadMaskedSpriteSheet(path,threshold=1,grey=False):
    sprites=[]
    sprite_name=getFilename(path)
    img_spr_sheet_raw=cv2.imread(path) 
    img_spr_sheet=clearBackground(img_spr_sheet_raw.copy(),[255,0,255])
    img_spr_sheet_grey=cv2.cvtColor(img_spr_sheet,cv2.COLOR_BGR2GRAY)
    img_spr_sheet_bin=greyToBinary(img_spr_sheet_grey,threshold)
    img_spr_sheet_bin=morphologicalTransformation(img_spr_sheet_bin,kernel_size=3)
    img_spr_sheet_mask=255-img_spr_sheet_bin.copy()
    contours=getContoursOfImage(img_spr_sheet_bin,show_edges=False)
    recs=contoursToRectangles(contours)
    print('Contours before simplify: {}'.format(len(recs)))
    recs=filterSmallRecs(recs,100,10,10)
    recs=simplifyOverlappingRectangles(recs)
    recs=filterSmallRecs(recs,250,20,20)
    print('Contours after simplify: {}'.format(len(recs)))
    if grey:
        img_spr_sheet_to_cut=cv2.cvtColor(img_spr_sheet_raw,cv2.COLOR_BGR2GRAY)
    else:
        img_spr_sheet_to_cut=img_spr_sheet_raw
        img_spr_sheet_mask=cv2.cvtColor(img_spr_sheet_mask,cv2.COLOR_GRAY2BGR)
    for rec in recs:
        rec=enlargeRec(rec,4)
        x0,y0,x1,y1=rec['x0'],rec['y0'],rec['x1'],rec['y1']
        img_single_sprite=img_spr_sheet_to_cut[y0:y1,x0:x1].copy()
        img_single_sprite_mask=img_spr_sheet_mask[y0:y1,x0:x1].copy()
        sprites.append({'spr':img_single_sprite,'mask':img_single_sprite_mask})
    return sprites


def loadAllKillerInstinctSpriteSheets(sprite_sheets=['Cinder','Combo','Eyedol','Fulgore','Glacius','Jago','Orchid','Riptor','Sabrewulf','Spinal','Thunder'],grey=False):
    sprites={}
    base_path='../games/sprites/killer_instinct/'
    extension='.png'
    for sprite_sheet in sprite_sheets:
        path=base_path+sprite_sheet+extension
        # sprites[sprite_sheet]=loadSpriteSheet(path,grey=grey)
        sprites[sprite_sheet]=loadMaskedSpriteSheet(path,grey=grey)
    return sprites

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def storeSprites(sprites,path,compress=True):
    data=json.dumps(sprites,cls=NumpyEncoder)
    if compress:
        compressed=zlib.compress(data.encode('utf-8'))
        with open(path,'wb') as file:
            file.write(compressed)
    else:
        with codecs.open(path, 'w', 'utf-8') as file:
            file.write(data)


def loadSprites(path,compress=True):
    data=None
    if compress:
        with open(path,'rb') as file:
            compressed=file.read()
            data=zlib.decompress(compressed).decode('utf-8')
    else:
        with codecs.open(path, 'r', 'utf-8', errors='ignore') as file:
            data=file.read()
    if data is not None:
        sprites_vanilla=json.loads(data)
        sprites={}
        for k,v in sprites_vanilla.items():
            sprite_list=[]
            for sprite in v:
                if type(sprite) is dict:
                    new_dict={}
                    for k2,v2 in sprite.items():
                        new_dict[k2]=np.array(v2).astype(np.uint8)
                    sprite_list.append(new_dict)
                else:
                    sprite_list.append(np.array(sprite).astype(np.uint8))
            sprites[k]=sprite_list
        return sprites