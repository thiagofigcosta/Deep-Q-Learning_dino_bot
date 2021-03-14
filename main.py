#!/bin/python

import sprite_reader as spriter
import screen_reader as screen_io
import sys,cv2,os

def main(argv):
    cache_sprites_path='killer_instinct_cutted_sprites.json.z'
    if not os.path.exists(cache_sprites_path):
        sprites=spriter.loadAllKillerInstinctSpriteSheets()
        spriter.storeSprites(sprites,cache_sprites_path)
    else:
        sprites=spriter.loadSprites(cache_sprites_path)

    for character,sprites in sprites.items():
        for sprite in sprites:
            cv2.imshow('cutted sprite',sprite)
            cv2.waitKey()
    
    # emulator_window_rec=screen_io.setup()
    # emulator_window=screen_io.captureScreen(emulator_window_rec)
    # emulator_window_name='emulator'
    # cv2.namedWindow(emulator_window_name)
    # cv2.imshow(emulator_window_name,emulator_window)
    # cv2.moveWindow(emulator_window_name,emulator_window_rec['x1']+10,emulator_window_rec['y0']) 
    # cv2.waitKey()

if __name__=='__main__':
    main(sys.argv[1:])