#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess

#opens the game 

class AIOpener(object):
	def __init__(self, path='/games/flappy.swf',opener='flashplayer',args=''):
		self.path=path
		self.opener=opener
		self.args=args
		
	def open(self):
		if self.path!='' and self.opener!='':
			self.process=subprocess.call(self.opener+' \"'+self.path+'\" '+self.args+' &', shell=True)

	def close(self):
		pass #closes the game