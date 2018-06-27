#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Ai.opener import AIOpener
from Ai.hands import Hands
from Ai.setupenv import Screen, Point
import time
#main class 

class FlappyBird(object):
	def __init__(self, online=True, jar=False, swf=False):
		self.game=None
		self.screen=None
		self.hands=Hands()
		if jar:
			self.gametype='java'
		else:
			self.gametype='flash'
		if online==False:
			if jar:
				self.game=AIOpener(path='/games/flappy.jar',opener='java -jar')
				time.sleep(.100)
				self.screen=Screen(Point(443,59),Point(923,59),Point(443,669),Point(923,669))
			elif swf:
				self.game=AIOpener(path='/games/flappy.swf',opener='flashplayer')
				time.sleep(.100)
				self.screen=Screen(Point(0,0),Point(0,0),Point(0,0),Point(0,0))
			else:
				print('Error, invalid game type. Game must be online, java or flash(swf)')
				raise SystemExit()
		else:
			time.sleep(2)
			self.screen=Screen(Point(456,115),Point(888,115),Point(888,719),Point(456,719))
		hands.putCursorOnOrigin(screen)

	def start(self):
		hands.focousScreen(screen)
		if self.game=='flash':
			hands.putCursorOnOrigin(screen)
			hands.moveMouse(125,435,rel=True) #okbutton
			hands.clickMouse()
			hands.moveMouse(0,50,rel=True) #startbutton
			hands.clickMouse()	
			hands.pressKeyboard('espace')
		elif self.game=='java':
			hands.pressKeyboard('espace')
			hands.clickMouse()

	def jump(self):
		if self.game=='flash':
			hands.pressKeyboard('espace')
		elif self.game=='java':
			hands.clickMouse()	


	def gameover(self,frame):
		if self.game=='flash':
			pass
		elif self.game=='java':
			pass

	def play(self,times=10):
		for i in range(times):
			frame=eyes.getScreen(screen)
			while not self.gameover(frame):
				action=brain.process(frame)
				if action=='jump':
					self.jump()
				frame=eyes.getScreen(screen)