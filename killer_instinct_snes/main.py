#!/bin/python

import sprite_reader as spriter
import screen_reader as screen_io
import sys,cv2,os,time
import numpy as np

def setup(grey=False,ignore_screen_capture=False):
    cache_sprites_path='killer_instinct_cutted_sprites.json.z'
    if not os.path.exists(cache_sprites_path):
        print('Loading sprite sheets and cutting sprites...')
        sprites=spriter.loadAllKillerInstinctSpriteSheets(['Cinder','Jago'],grey=grey)
        spriter.storeSprites(sprites,cache_sprites_path)
        print('Loading sprite sheets and cutting sprites...OK')
    else:
        print('Loading sprites...', end='')
        sprites=spriter.loadSprites(cache_sprites_path)
        print('OK')
    if not ignore_screen_capture:
        print('Selecting screen window to detect the game...')
        emulator_window_rec=screen_io.setup()
        print('Selecting screen window to detect the game...OK')
    else:
        emulator_window_rec=None
    return sprites, emulator_window_rec


def ingameLoop(killer_instinct_cutted_sprites,emulator_window_rec,grey=False, display=False):
    ingame=True
    fps=24
    emulator_window_name='emulator'
    cv2.namedWindow(emulator_window_name)
    cv2.moveWindow(emulator_window_name,emulator_window_rec['x1']+10,emulator_window_rec['y0'])
    ms_p_f=1000/fps
    while(ingame):
        start=time.time()
        # start loop
        frame_game=screen_io.captureScreen(emulator_window_rec,grey=grey)
        characters_in_game=['Cinder','Jago']
        found_elements=findElementsOnScreen(frame_game,killer_instinct_cutted_sprites,characters_in_game,stop_on_first=True)
        
        if display:
            if grey:
                frame_game_bgr=cv2.cvtColor(frame_game,cv2.COLOR_GRAY2BGR)
            else:
                frame_game_bgr=frame_game
            thickness=1
            color=(0,255,0)
            for el in found_elements:
                rec=el['rec']
                x0,y0,x1,y1=rec['x0'],rec['y0'],rec['x1'],rec['y1']
                cv2.rectangle(frame_game_bgr,(x0,y0),(x1,y1),color,thickness)
            cv2.imshow(emulator_window_name,frame_game_bgr)
            if cv2.waitKey(1)==27: #escape key
                break
            # end loop
            # # respect fps
            # delta=int(ms_p_f-(time.time()-start))
            # if delta<=0:
            #     delta=1
            # if cv2.waitKey(delta)==27: #escape key
            #     break

def showSprites(sprs):
    for character,sprites in sprs.items():
        for i in range(len(sprites)):
            if type(sprites[i]) is dict:
                sprite=sprites[i]['spr']
                mask=sprites[i]['mask']
                sprite=np.hstack((sprite, mask))
            else:
                sprite=sprites[i]
                mask=None
            cv2.imshow('{}-{}'.format(character,i),sprite)
        cv2.waitKey()


def findElementsOnScreen(screen,to_match_sprites,sprite_names_to_search,stop_on_first=False):
    screen=screen.astype(np.uint8)
    img_match_threshold=0.8
    img_match_min_diff=10
    found_elements=[]
    for character,sprites in to_match_sprites.items():
        if character in sprite_names_to_search:
            candidate=None
            candidate_val=None
            candidate_index=None
            for i in range(len(sprites)):
                if type(sprites[i]) is dict:
                    sprite=sprites[i]['spr']
                    mask=sprites[i]['mask']
                else:
                    sprite=sprites[i]
                    mask=None
                sprite_w,sprite_h=sprite.shape[1],sprite.shape[0]
                if mask is None:
                    method=cv2.TM_CCOEFF_NORMED
                    res=cv2.matchTemplate(screen,sprite,method) 
                else:
                    method=cv2.TM_SQDIFF # TM_SQDIFF or TM_CCORR_NORMED for mask
                    res=cv2.matchTemplate(screen,sprite,method,mask=mask) 
                # only the best
                min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)
                if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    search_min=True
                    loc=min_loc
                    val=min_val
                else:
                    search_min=False
                    loc=max_loc
                    val=max_val
                if  (val>=img_match_threshold and not search_min) or (val<=img_match_min_diff and search_min):
                    if candidate is None or (val>candidate_val and not search_min) or (val<candidate_val and search_min):
                        rec=spriter.pointAndSizeToRectangle(loc[0],loc[1],sprite_w,sprite_h)
                        candidate=rec
                        candidate_val=val
                        candidate_index=i
                        if stop_on_first:
                            break
            if candidate is not None:
                found_elements.append({'el':character,'rec':candidate,'idx':candidate_index})
    return found_elements            

def testMatching(grey):
    scene_path='../games/sprites/killer_instinct/ingame_0.png'
    if grey:
        scene=cv2.imread(scene_path, cv2.IMREAD_GRAYSCALE)
    else:
        scene=cv2.imread(scene_path)
    sprs,_=setup(grey=grey,ignore_screen_capture=True)
    elements=findElementsOnScreen(scene,sprs,['Cinder','Jago'])
    if grey:
        scene_bgr=cv2.cvtColor(scene,cv2.COLOR_GRAY2BGR)
    else:
        scene_bgr=scene
    print('Found {} elements'.format(len(elements)))
    thickness=1
    color=(0,255,0)
    for el in elements:
        rec=el['rec']
        x0,y0,x1,y1=rec['x0'],rec['y0'],rec['x1'],rec['y1']
        cv2.rectangle(scene_bgr,(x0,y0),(x1,y1),color,thickness)
    cv2.imshow('Screen',scene_bgr)
        
def main(argv):
    grey=True
    sprs,rec=setup(grey=grey) 
    ingameLoop(sprs,rec,grey=grey)   
    # showSprites(sprs)
    # testMatching(grey)
    cv2.waitKey()

if __name__=='__main__':
    main(sys.argv[1:])