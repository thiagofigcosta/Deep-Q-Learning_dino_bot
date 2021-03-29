#!/bin/python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # DISABLE TENSORFLOW WARNING
import sys,cv2,time,pyautogui,json,zlib,keras,codecs
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer, Dense

def queryYesOrNo(question, default='yes'):
    # default == 'yes' or 'no' or None
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)
    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def pointsToRectangle(x0,y0,x1,y1):
    w=int(x1-x0)
    h=int(y1-y0)
    rec={'x0':int(x0),'x1':int(x1),'y0':int(y0),'y1':int(y1),'w':int(w),'h':int(h)}
    return rec

def pointAndSizeToRectangle(x,y,w,h):
    rec={'x0':int(x),'x1':int(x+w),'y0':int(y),'y1':int(y+h),'w':int(w),'h':int(h)}
    return rec

def getRectangleCenter(rec):
    return {'x':int(rec['x0']+rec['w']/2),'y':int(rec['y0']+rec['h']/2)}

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

def getEquivalentRectangles(rec_0,rec_1,offset=0):
    equivalent=[]
    collision=checkIfRectanglesIntersects(rec_0,rec_1,offset=offset)
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
    return equivalent


def simplifyOverlappingRectangles(recs,offset=0):
    final_recs=[]
    already_simplified_indexes=set()
    for i,rec_0 in enumerate(recs):
        if i not in already_simplified_indexes:
            go_again=True
            while go_again:
                go_again=False
                for j,rec_1 in enumerate(recs):
                    if j>i and j not in already_simplified_indexes:
                        eq=getEquivalentRectangles(rec_0,rec_1,offset=offset)
                        if len(eq)==1:
                            rec_0=eq[0]
                            already_simplified_indexes.add(j)
                            go_again=True
                            break
            final_recs.append(rec_0)
    return final_recs

def simplifyOverlappingCactus(cactus,offset=0):
    final_recs=[]
    already_simplified_indexes=set()
    for i,cactus_0 in enumerate(cactus):
        if i not in already_simplified_indexes:
            go_again=True
            while go_again:
                go_again=False
                for j,cactus_1 in enumerate(cactus):
                    if j>i and j not in already_simplified_indexes:
                        eq=getEquivalentRectangles(cactus_0['rect'],cactus_1['rect'],offset=offset)
                        if len(eq)==1:
                            cactus_0['rect']=eq[0]
                            cactus_0['trust']=(cactus_0['trust']+cactus_1['trust'])/2
                            cactus_0['idx']=-1
                            already_simplified_indexes.add(j)
                            go_again=True
                            break
            final_recs.append(cactus_0)
    return final_recs

def getGameScreenBoundaries():
    def draw_rect(event,x,y,flags,param):
        nonlocal x0,y0,x1,y1,drawing,screen_bkp,screen,selected_square,thickness,color
        if event==cv2.EVENT_LBUTTONDOWN:
            drawing=True
            x0,y0=x,y
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


def captureScreen(area_rec):
    screen=np.array(pyautogui.screenshot())
    screen=screen[area_rec['y0']:area_rec['y1'],area_rec['x0']:area_rec['x1']]
    screen=cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    return screen 

def saveJsonToFile(json_obj,path,compress=False):
    data=json.dumps(json_obj)
    if compress:
        compressed=zlib.compress(data.encode('utf-8'))
        with open(path,'wb') as file:
            file.write(compressed)
    else:
        with codecs.open(path, 'w', 'utf-8') as file:
            file.write(data)

def loadJsonFromFile(path,compress=False):
    data=None
    if compress:
        with open(path,'rb') as file:
            compressed=file.read()
            data=zlib.decompress(compressed).decode('utf-8')
    else:
        with codecs.open(path, 'r', 'utf-8', errors='ignore') as file:
            data=file.read()
    if data is not None:
        data=json.loads(data)
    return data

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

def loadSprites(path):
    img_sprite=cv2.imread(path) 
    img_sprite_no_background=clearBackground(img_sprite.copy(),[255,0,255])
    img_sprite=cv2.cvtColor(img_sprite,cv2.COLOR_BGR2GRAY)
    img_sprite_no_background=cv2.cvtColor(img_sprite_no_background,cv2.COLOR_BGR2GRAY)
    img_sprite_no_background=greyToBinary(img_sprite_no_background,1)
    img_sprite_mask=255-img_sprite_no_background
    return {'sprite':img_sprite,'mask':img_sprite_mask}

def loadAssets(base_path='games/sprites/dino/'):
    print('Loading assets...', end='')
    asset_list=['bird_1','bird_2','cactus_big_large','cactus_big_thin','cactus_regular','cactus_small','dino_down','dino_up','game_over','ground','hi','number_0','number_1','number_2','number_3','number_4','number_5','number_6','number_7','number_8','number_9']
    ext='.png'
    loaded_assets={}
    for asset in asset_list:
        path='{}{}{}'.format(base_path,asset,ext)
        sprites=loadSprites(path)
        loaded_assets[asset]=sprites
    organized_assets={'bird':[],'cactus':[],'dino':[],'numbers':list(range(10))}
    for k,v in loaded_assets.items():
        if k.startswith('bird'):
            organized_assets['bird'].append(v)
        elif k.startswith('cactus'):
            organized_assets['cactus'].append(v)
        elif k.startswith('dino'):
            organized_assets['dino'].append(v)
        elif k.startswith('number'):
            organized_assets['numbers'][int(k.split('_')[1])]=v
        else:
            organized_assets[k]=[v]
    print('OK')
    return organized_assets

def matchSprites(screen,template_list,find_all,stop_on_first=True,sensitivity=(0.9,1),x_offset=0,y_offset=0):
    img_match_threshold=sensitivity[0]
    img_match_min_diff=sensitivity[1]
    found_elements=[]
    candidate=None
    candidate_val=None
    candidate_index=None
    for i in range(len(template_list)):
        if type(template_list[i]) is dict:
            if 'mask' in template_list[i]:
                sprite=template_list[i]['sprite']
                mask=template_list[i]['mask']
            else:
                sprite=template_list[i]['sprite']
                mask=None
        else:
            sprite=template_list[i]
            mask=None
        sprite_w,sprite_h=sprite.shape[1],sprite.shape[0]
        if mask is None:
            method=cv2.TM_CCOEFF_NORMED
            res=cv2.matchTemplate(screen,sprite,method) 
        else:
            method=cv2.TM_SQDIFF # TM_SQDIFF or TM_CCORR_NORMED for mask
            res=cv2.matchTemplate(screen,sprite,method,mask=mask) 
        search_min=(method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED])

        if find_all:
            loc=np.where(np.logical_or(np.logical_and(np.greater_equal(res,img_match_threshold),np.logical_not(search_min)),np.logical_and(np.less_equal(res,img_match_min_diff),search_min))) # (res>=img_match_threshold and not search_min) or (res<=img_match_min_diff and search_min)
            filtered_locations=zip(*loc[::-1])
            for pt in filtered_locations:
                val=0 # TODO implement me, I want to get the points and their values
                if search_min:
                    val=(255-val)/255
                rec=pointAndSizeToRectangle(pt[0]+x_offset,pt[1]+y_offset,sprite_w,sprite_h)
                found_elements.append({'rect':rec,'idx':i,'trust':val})
        else:
            min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)
            if search_min:
                loc=min_loc
                val=min_val
            else:
                loc=max_loc
                val=max_val
            if (val>=img_match_threshold and not search_min) or (val<=img_match_min_diff and search_min):
                if candidate is None or (val>candidate_val and not search_min) or (val<candidate_val and search_min):
                    rec=pointAndSizeToRectangle(loc[0]+x_offset,loc[1]+y_offset,sprite_w,sprite_h)
                    candidate=rec
                    candidate_val=val
                    candidate_index=i
                    if stop_on_first:
                        break
    if not find_all and candidate is not None:
        if search_min:
            candidate_val=(255-candidate_val)/255
        found_elements.append({'rect':candidate,'idx':candidate_index,'trust':candidate_val})
    return found_elements

def setup(ignore_screenshot=False):
    assets=loadAssets()
    game_boundaries_rect=None
    if not ignore_screenshot:
        game_boundaries_rect=getGameScreenBoundaries()
    return assets,game_boundaries_rect

def drawRectsOnScene(scene_bgr,found_sprites):
    rect_thickness=1
    rect_color=(0,255,0)
    for el in found_sprites:
        rec=el['rect']
        x0,y0,x1,y1=rec['x0'],rec['y0'],rec['x1'],rec['y1']
        cv2.rectangle(scene_bgr,(x0,y0),(x1,y1),rect_color,rect_thickness)
    return scene_bgr

def parseAndFilterScore(hi,numbers):
    if len(numbers)<1:
        return 0, numbers
    if type(hi) is list:
        if len(hi)<1:
            return 0, numbers
        hi=hi[0]
    hi_x1=hi['rect']['x1']
    number_size=numbers[0]['rect']['x1']-numbers[0]['rect']['x0']
    cur_score_start_pos=hi_x1+7*number_size
    numbers=[n for n in numbers if n['rect']['x0']>cur_score_start_pos]
    numbers=sorted(numbers,key=lambda k:k['rect']['x0'],reverse=True) 
    score=0
    for i,number in enumerate(numbers):
        score+=10**i*number['idx']
    return score, numbers

def getAIMaximumValues():
    # TODO refine values
    return {'no_hdist':555,'no_vdist':59,'no_w':74,'no_h':54,'speed':1000,'dino_y':93}

def getAIDefaultValues():
    return {'no_hdist':0,'no_vdist':7,'speed':394.1534831108713,'dino_y':15,'ground_y':135}

def parseAiValues(AI):
    out=[]
    out.append(AI['no_hdist'])
    out.append(AI['no_vdist'])
    out.append(AI['no_w'])
    out.append(AI['no_h'])
    out.append(AI['speed'])
    out.append(AI['dino_y'])
    return out   

def normalizeAiValues(AI,check_out_of_range=True):
    max_values=getAIMaximumValues()
    out=[]
    out.append(AI['no_hdist']/max_values['no_hdist'])
    out.append(AI['no_vdist']/max_values['no_vdist'])
    out.append(AI['no_w']/max_values['no_w'])
    out.append(AI['no_h']/max_values['no_h'])
    out.append(AI['speed']/max_values['speed'])
    out.append(AI['dino_y']/max_values['dino_y'])
    if check_out_of_range:
        for i,el in enumerate(out):
            if el<0 or el>1:
                if el>1 or i!=0: # hdist can be negative
                    print ('ERROR: element({}) at index {} out of range!'.format(el,i))
    return out

def parseFrame(scene,assets,context=None,subtract_default_inputs=True):
    cur_time=time.time()
    default_AI_values=getAIDefaultValues()
    if not subtract_default_inputs:
        for k,_ in default_AI_values.items():
            if k!='ground_y':
                default_AI_values[k]=0
    # color check
    scene_middle_x=int(scene.shape[1]/2)
    scene_middle_y=int(scene.shape[0]/2)
    if scene[scene_middle_y][scene_middle_x]==0: # scene colors are inverted
        scene=255-scene
    # ground
    ground_rect=pointAndSizeToRectangle(0,0,int(scene.shape[1]*.1),scene.shape[0])
    scene_left=scene[ground_rect['y0']:ground_rect['y1'],ground_rect['x0']:ground_rect['x1']]
    ground=matchSprites(scene_left,assets['ground'],find_all=False,sensitivity=(1,0))
    if len(ground)==1:
        ground_y=ground[0]['rect']['y0']
    else:
        ground_y=default_AI_values['ground_y']
    # dino
    dino=matchSprites(scene,assets['dino'],find_all=False,sensitivity=(0.7,20))
    has_dino=len(dino)==1
    if has_dino:
        dino_pos=getRectangleCenter(dino[0]['rect'])
        dino_rect=dino[0]['rect']
    else:
        dino_pos={'x':None,'y':None}
        dino_rect=None
    # cactus
    cactus=matchSprites(scene,assets['cactus'],find_all=True,sensitivity=(0.8,10))
    cactus=simplifyOverlappingCactus(cactus,offset=2)
    amount_cactus=len(cactus)
    # bird
    birds=matchSprites(scene,assets['bird'],find_all=True,sensitivity=(0.8,10))
    amount_birds=len(birds)
    # numbers
    x_offset=int(scene.shape[1]/2)
    score_rect=pointAndSizeToRectangle(x_offset,0,scene.shape[1],int(scene.shape[0]*.18))
    scene_upper=scene[score_rect['y0']:score_rect['y1'],score_rect['x0']:score_rect['x1']]
    numbers=matchSprites(scene_upper,assets['numbers'],find_all=True,x_offset=x_offset,sensitivity=(0.95,1))
    hi=matchSprites(scene_upper,assets['hi'],find_all=False,x_offset=x_offset,sensitivity=(0.8,10))
    score,numbers=parseAndFilterScore(hi,numbers)
    if context is not None:
        score=max(context['last_score'],score)
    has_hi=len(hi)==1
    amount_numbers=len(numbers)
    # gg
    gg=matchSprites(scene,assets['game_over'],find_all=False,sensitivity=(0.8,10))
    has_gg=len(gg)==1
    
    # AI inputs 
    next_obstacle=None
    if dino_rect is not None:
        next_obstacle_x=scene.shape[1]
        for c in cactus:
            if c['rect']['x0']<next_obstacle_x and c['rect']['x0'] > dino_rect['x0']:
                next_obstacle=c['rect']
                next_obstacle_x=next_obstacle['x0']
        for b in birds:
            if b['rect']['x0']<next_obstacle_x and b['rect']['x0'] > dino_rect['x0']:
                next_obstacle=b['rect']
                next_obstacle_x=next_obstacle['x0']
        dino_y=max(ground_y-dino_pos['y']-default_AI_values['dino_y'],0)
    else:
        dino_y=0 
    if next_obstacle is None or dino_rect is None:
        max_AI_values=getAIMaximumValues()
        next_obstacle_pos={'x':None,'y':None}
        next_obstacle_hdistance=max_AI_values['no_hdist']
        next_obstacle_vdistance=max_AI_values['no_vdist']
        next_obstacle_weight=0
        next_obstacle_height=0 
        if context is not None:
            context['passed_obstacle']=False
            context['already_passed_obstacle']=False
            speed=context['last_speed']
        else:
            speed=0
    else:
        next_obstacle_pos=getRectangleCenter(next_obstacle)
        next_obstacle_hdistance=next_obstacle['x0']-dino_pos['x'] #max(next_obstacle_pos['x']-dino_rect['x1'],0)
        next_obstacle_vdistance=max(ground_y-next_obstacle['y0']-default_AI_values['no_vdist'],0)
        next_obstacle_weight=next_obstacle['w']
        next_obstacle_height=next_obstacle['h']
        speed=0
        if context is not None:
            if context['last_time'] is not None and context['last_no_pos_x'] is not None:
                if (dino_rect['x1']>next_obstacle_pos['x'] or next_obstacle_pos['x']>context['last_no_pos_x']) and not context['already_passed_obstacle']:
                    context['passed_obstacle']=True
                    context['already_passed_obstacle']=True
                else:
                    context['passed_obstacle']=False
                    if context['last_speed'] is not None:
                        l_speed=context['last_speed']
                    else:
                        l_speed=-1
                    speed=max((context['last_no_pos_x']-next_obstacle_pos['x'])/(cur_time-context['last_time'])-default_AI_values['speed'],0,l_speed) # pixels per second
                    # speed=max((context['last_no_pos_x']-next_obstacle_pos['x']),0,l_speed) # pixels per frame unit
                    if l_speed>0 and l_speed*2<speed: # avoid super speed increase
                        speed=l_speed
            elif context['last_speed'] is not None:
                speed=context['last_speed']
            if context['last_no_pos_x'] is not None and next_obstacle_pos['x']>context['last_no_pos_x']: # new obstable
                context['already_passed_obstacle']=False

    return {'dino':has_dino,'amount_cactus':amount_cactus,'amount_birds':amount_birds,'score':score,'amount_numbers':amount_numbers,'game_is_over':has_gg,'cur_time':cur_time,
            'matches':{'dino':dino,'cactus':cactus,'bird':birds,'numbers':numbers,'gg':gg,'hi':hi,'ground':ground},
            'positions':{'dino':dino_pos,'no_pos':next_obstacle_pos},
            'AI':{'no_hdist':next_obstacle_hdistance,'no_vdist':next_obstacle_vdistance,'no_w':next_obstacle_weight,'no_h':next_obstacle_height,'speed':speed,'dino_y':dino_y}}

def test():
    assets=setup(ignore_screenshot=True)[0]
    scene_paths=['games/sprites/dino/screenshots/bird.png','games/sprites/dino/screenshots/bird_high.png','games/sprites/dino/screenshots/game_over.png','games/sprites/dino/screenshots/inverted_screen.png','games/sprites/dino/screenshots/nested_obstacles.png','games/sprites/dino/screenshots/no_obstacles.png']
    for i,path in enumerate(scene_paths):
        scene=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        scene_parsed=parseFrame(scene,assets)
        # print
        print('Scene {}'.format(i))
        print('\tScore: {}'.format(scene_parsed['score']))
        if scene_parsed['dino']:
            print('\tFound dino at x:{} y:{}'.format(scene_parsed['positions']['dino']['x'],scene_parsed['positions']['dino']['y']))
        else:
            print('\tDino not found')
        print('\tFound {} cactus'.format(scene_parsed['amount_cactus']))
        print('\tFound {} birds'.format(scene_parsed['amount_birds']))
        print('\tFound {} numbers'.format(scene_parsed['amount_numbers']))
        if scene_parsed['game_is_over']:
            print('\tGame over')
        else:
            print('\tAI inputs:')
            for k,v in scene_parsed['AI'].items():
                print('\t\t{}: {}'.format(k,v))
        # draw
        to_show=cv2.cvtColor(scene,cv2.COLOR_GRAY2BGR) # just to draw boundaries
        drawRectsOnScene(to_show,scene_parsed['matches']['dino'])
        drawRectsOnScene(to_show,scene_parsed['matches']['cactus'])
        drawRectsOnScene(to_show,scene_parsed['matches']['numbers'])
        drawRectsOnScene(to_show,scene_parsed['matches']['bird'])
        drawRectsOnScene(to_show,scene_parsed['matches']['gg'])
        drawRectsOnScene(to_show,scene_parsed['matches']['hi'])
        drawRectsOnScene(to_show,scene_parsed['matches']['ground'])
        cv2.imshow('Scene {}'.format(i),to_show)
    cv2.waitKey()

def updateContext(context,parsed_scene):
    context['last_time']=parsed_scene['cur_time']
    context['last_speed']=parsed_scene['AI']['speed']
    context['last_no_pos_x']=parsed_scene['positions']['no_pos']['x']
    return context

def getFreshContext():
    return {'last_time':None,'last_no_pos_x':None,'last_speed':0,'last_score':0,'took_action':-1,'last_state':[],'last_actions':[],'passed_obstacle':False,'already_passed_obstacle':False}

def performAction(action):
    if action=='jump':
        pyautogui.keyUp('down')
        pyautogui.press('up')
    elif action=='down':
        pyautogui.keyDown('down')
        # pyautogui.press('down')
    else: # action==stay
        pyautogui.keyUp('down')

def performIntAction(action,actions_list):
    action_str=actions_list[action]
    performAction(action_str)
    return action_str

def floatListToFormatedStr(f_list):
    names=['hd','vd','w','h','vel','d_h']
    str_formatted='['
    for i,el in enumerate(f_list):
        str_formatted+=' {}:{:.3f}'.format(names[i],el)
    return str_formatted+' ]'

def saveModel(model_weights_path,model_metadata_path,cur_episode,epsilon,max_scores,neural_network):
    print('Saving model...',end='')
    metadata={}
    metadata['cur_episode']=cur_episode
    metadata['epsilon']=epsilon
    metadata['max_scores']=max_scores
    saveJsonToFile(metadata,model_metadata_path)
    neural_network.save(model_weights_path)
    metadata={}
    print('OK')

def getGameFocus(game_window_rec):
    window_pos=getRectangleCenter(game_window_rec)
    mouse_pos_backup=pyautogui.position()
    pyautogui.click(window_pos['x']-50, window_pos['y'], button='left') # get focus
    pyautogui.moveTo(mouse_pos_backup[0],mouse_pos_backup[1])
    time.sleep(0.666)

def thresholdAction(state,actions,normalized=True):
    action='stay'
    h_dist=state[0]
    v_dist=state[1]
    w=state[2]
    h=state[3]
    vel=state[4]
    dino_h=state[5]
    # TODO compute manually action based on the state
    # TODO normalize
    if h_dist<220:
        action='jump'
    elif v_dist>h:
        action='down'
    return actions.index(action)

def playDino(assets,game_window_rec,limit_fps=30,display=False,show_speeds=False,load_model=True,save_model=True,learn=True,verbose=False,episodes_frequency_to_save=20,episodes_frequency_to_reload=300):
    ingame=True
    max_fps_warnings=30
    if display:
        emulator_window_name='game'
        cv2.namedWindow(emulator_window_name)
        cv2.moveWindow(emulator_window_name,game_window_rec['x0'],game_window_rec['y1']+66*2)
    if limit_fps!=0:
        s_p_f=1/limit_fps
    context=getFreshContext()
    speeds=set()
    # AI settings start 
    actions=('jump','down','stay')
    use_AI=True
    # NN
    amount_AI_inputs=6
    learning_rate=0.007
    hidden_neurons=6
    action_function='relu'
    use_bias=True
    normalize_inputs=False
    sleep_after_action=0
    subtract_default_inputs=normalize_inputs
    # Q-learning
    frames_to_consider_effect=2
    lose_game_penalty=100
    default_behaviour_penalty=0
    pass_obstacle_reward=10
    stay_alive_reward=0
    fixed_reward_instead_of_score_based=True
    discount_factor=0.95
    epsilon_max=1.0
    epsilon_min=0.101
    epsilon_decay=0.03
    epsilon_confidance=0.1 # stops random shots when below this value
    stay_alive_reward+=default_behaviour_penalty
    # persistance
    model_weights_path='bot_brain.h5'
    model_metadata_path='bot_metadata.json'
    # AI settings end 
    if load_model and os.path.isfile(model_weights_path):
        print('Loading model...',end='')
        neural_network=keras.models.load_model(model_weights_path)
        print('OK')
    else:
        print('Creating model...',end='')
        neural_network=Sequential()
        neural_network.add(InputLayer(batch_input_shape=(1,amount_AI_inputs)))
        neural_network.add(Dense(hidden_neurons,activation=action_function,use_bias=use_bias))
        neural_network.add(Dense(len(actions),activation=action_function,use_bias=use_bias))
        neural_network.compile(loss='mse',optimizer=keras.optimizers.Adam(learning_rate=learning_rate),metrics=['mae']) # mae = mean absolute error, mse = mean squared error
        print('OK')
    if load_model and os.path.isfile(model_metadata_path):
        print('Loading metadata...',end='')
        metadata=loadJsonFromFile(model_metadata_path)
        cur_episode=metadata['cur_episode']
        epsilon=metadata['epsilon']
        max_scores=metadata['max_scores']
        metadata={}
        print('OK')
    else:
        cur_episode=0
        epsilon=1.0
        max_scores=[]   
    action=0
    reward=0
    cur_fps_warnings=0
    getGameFocus(game_window_rec)
    performAction('jump')
    while(ingame):
        try:
            start_time=time.time()
            # screen read and parse
            if context['took_action']!=0 or context['took_action']<0:
                if context['took_action']<0:
                    context['took_action']=0
                game_frame=captureScreen(game_window_rec)
                parsed_frame=parseFrame(game_frame,assets,context=context,subtract_default_inputs=subtract_default_inputs) 
                last_score=context['last_score']
                context=updateContext(context,parsed_frame)
                if normalize_inputs:
                    state=normalizeAiValues(parsed_frame['AI'])
                else:
                    state=parseAiValues(parsed_frame['AI'])
                if verbose and context['passed_obstacle'] and reward==0:
                    print('Passed Obstacle! Took action: {}'.format(context['took_action']))
            # AI
            if parsed_frame['game_is_over']:
                reward=-lose_game_penalty
            elif context['took_action']==0:
                if use_AI:
                    if epsilon>=epsilon_confidance and np.random.random()<=epsilon:
                        action=np.random.randint(0,len(actions))
                        context['last_actions']=[]
                    else:
                        pred_actions=neural_network.predict([state])
                        context['last_actions']=pred_actions[0].tolist()
                        action=np.argmax(pred_actions)
                else:
                    action=thresholdAction(state,actions,normalized=normalize_inputs)
                    context['last_actions']=[]
                reward=0
                performIntAction(action,actions)
                context['last_state']=state
                time.sleep(sleep_after_action)
            else:
                if context['took_action']==frames_to_consider_effect:
                    reward+=stay_alive_reward if fixed_reward_instead_of_score_based else (parsed_frame['score']-last_score)
                    if actions[action]=='stay':
                        reward-=default_behaviour_penalty
                if context['passed_obstacle']:
                    reward=pass_obstacle_reward
            if (parsed_frame['game_is_over'] or context['took_action']==frames_to_consider_effect) and len(context['last_state'])>0 and learn:
                if len(context['last_actions'])==0:
                    previous_state_mirror_labels=neural_network.predict([context['last_state']])[0].tolist()
                else:
                    previous_state_mirror_labels=context['last_actions']
                ajusted_label_for_action=reward+discount_factor*np.max(neural_network.predict([state]))
                previous_state_mirror_labels[action]=ajusted_label_for_action
                neural_network.fit([context['last_state']],[previous_state_mirror_labels],epochs=1,verbose=0)
                invert_action_state=True
                if (verbose):
                    print('Action: {} - Reward: {:4} | E: {:.3f} | state: {}'.format(actions[action],reward,epsilon,floatListToFormatedStr(context['last_state'])))
            if parsed_frame['game_is_over']:
                if len(context['last_state'])>0:
                    cur_episode+=1
                    max_scores.append(parsed_frame['score'])
                    epsilon=epsilon_min+(epsilon_max-epsilon_min)*np.exp(-epsilon_decay*cur_episode)
                    if save_model and cur_episode%episodes_frequency_to_save==0 and cur_episode>0:
                        saveModel(model_weights_path,model_metadata_path,cur_episode,epsilon,max_scores,neural_network)
                else: 
                    getGameFocus(game_window_rec)
                if cur_episode>0 and cur_episode%episodes_frequency_to_reload==0: # reload the game to avoid bugs
                    pyautogui.press('f5')
                    time.sleep(1.666)
                performAction('jump') # restart the game
                context=getFreshContext()
                time.sleep(0.666)
            else:
                context['took_action']+=1
                if context['took_action']>frames_to_consider_effect:
                    context['took_action']=0 if frames_to_consider_effect>0 else -1
            # display
            if show_speeds and parsed_frame['AI']['speed'] not in speeds:
                print('speed: {}'.format(parsed_frame['AI']['speed']))
                speeds.add(parsed_frame['AI']['speed'])
            if display:
                to_show=cv2.cvtColor(game_frame,cv2.COLOR_GRAY2BGR) # just to draw boundaries
                drawRectsOnScene(to_show,parsed_frame['matches']['dino'])
                drawRectsOnScene(to_show,parsed_frame['matches']['cactus'])
                drawRectsOnScene(to_show,parsed_frame['matches']['numbers'])
                drawRectsOnScene(to_show,parsed_frame['matches']['bird'])
                drawRectsOnScene(to_show,parsed_frame['matches']['gg'])
                drawRectsOnScene(to_show,parsed_frame['matches']['hi'])
                cv2.imshow(emulator_window_name,to_show)
            # end loop
            end_time=time.time()
            if limit_fps!=0 and context['took_action']!=0:
                # respect fps
                elapsed_seconds=float(end_time-start_time)
                s_to_wait=s_p_f-elapsed_seconds
                if s_to_wait<0:
                    if not parsed_frame['game_is_over'] and cur_fps_warnings<max_fps_warnings:
                        cur_fps_warnings+=1
                        print('WARNING: Low performance! Fixed fps: {} Real fps: {:.3f}'.format(limit_fps,(1/elapsed_seconds)))
                    s_to_wait=0
                if display:
                    ms_to_wait=int(s_to_wait*1000)
                    if ms_to_wait==0:
                        ms_to_wait=1
                    if cv2.waitKey(ms_to_wait)==27: #escape key
                        break
                else:
                    time.sleep(s_to_wait)
            elif display:
                if cv2.waitKey(1)==27: #escape key
                    break
        except KeyboardInterrupt:
            print('Caught ctrl+c')
            try:
                if save_model and queryYesOrNo('Do you wish to save the weigths?',default=None):
                    saveModel(model_weights_path,model_metadata_path,cur_episode,epsilon,max_scores,neural_network)
            except:
                pass
            sys.exit()
    if save_model:
        saveModel(model_weights_path,model_metadata_path,cur_episode,epsilon,max_scores,neural_network)

def main(argv):
    # test()
    assets,game_window_rec=setup()
    playDino(assets,game_window_rec,limit_fps=10,display=False,verbose=True)

if __name__=='__main__':
    main(sys.argv[1:])