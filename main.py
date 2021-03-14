#!/bin/python

import sprite_reader as spriter
import screen_reader as screen_io
import sys, cv2

def main(argv):
    # sprites=spriter.loadAllKillerInstinctSpriteSheets()
    emulator_window_rec=screen_io.setup()
    emulator_window=screen_io.captureScreen(emulator_window_rec)
    cv2.imshow('emulator',emulator_window)
    cv2.moveWindow('emulator',emulator_window_rec['x1']+10,emulator_window_rec['y0']) 
    cv2.waitKey()

if __name__=='__main__':
    main(sys.argv[1:])