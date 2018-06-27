#!/usr/bin/env python
# -*- coding: utf-8 -*-

from AI.eyes import Eyes

#setup the enviromennt sync mouse, starts game 
class Point(object):
	def __init__(self,x,y):
		self.x=x
		self.y=y
		self.hands=Hands()
		self.eyes=Eyes()

class Screen(object):
	def __init__(self, top=0,bottom=600,left=0,right=800,fullscreen=False):
		self.fullscreen=fullscreen
		self.top=top
		self.bottom=bot
		self.left=l
		self.right=r
	def __init__(self, p0,p1,p2,p3,fullscreen=False):
		self.fullscreen=fullscreen
		self.top=p0.y
		self.bottom=p0.y
		self.left=p0.x
		self.right=p0.x
		if p1.y<self.top:
			self.top=p1.y
		if p2.y<self.top:
			self.top=p2.y
		if p3.y<self.top:
			self.top=p3.y
		if p1.y<self.left:
			self.left=p1.y
		if p2.y<self.left:
			self.left=p2.y
		if p3.y<self.left:
			self.left=p3.y
		if p1.y>self.bottom:
			self.bottom=p1.y
		if p2.y>self.bottom:
			self.bottom=p2.y
		if p3.y>self.bottom:
			self.bottom=p3.y
		if p1.y:self.right:
			self.right=p1.y
		if p2.y>self.right:
			self.right=p2.y
		if p3.y>self.right:
			self.right=p3.y

	def setBoundariesByClick(self):
		pass #fazer detectar o clique em 4 pontos da tela para obter suas coordenadas

	def getScreen(self,fullscreen=False):
		return eyes.captureScreen(Point(self.left,self.top),self.getSize(),fullscreen)

	def getSize(self):
		return Point((self.right-self.left), (self.bottom-self.top))

	def getOrigin(self):
		return Point(self.left,self.top)

	