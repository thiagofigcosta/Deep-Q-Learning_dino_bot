#!/usr/bin/env python
# -*- coding: utf-8 -*-

#genetic and renural networks 


class MLP(object):
    def degrau(self,m):
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if m[i,j] >= 0:
                    m[i,j] = 1.0
                else:
                    m[i,j] = 0.0
        return m
    def evaluate(self,W, b, X, D):
        acc=0
        y=np.matrix(np.zeros(D.shape))
        for i in range(X.shape[0]):
            y[i]=self.degrau(np.add(np.dot(W,X[i].transpose()).transpose(),b))
            if np.all(y[i]==D[i]):
                acc=acc+1.0
        acc=(acc/X.shape[0])*100.0
        return acc, y
    def train(self,max_it, a, X, D):
        W=np.matrix(np.zeros(shape=(D.shape[1],X.shape[1])))
        b=np.zeros(D.shape[1])
        t=0
        E=1
        Err=np.zeros(max_it)
        while t<max_it and E>0:
            E=0
            for i in range(X.shape[0]):
                y=self.degrau(np.add(np.dot(W,X[i].transpose()).transpose(),b))
                e=np.add(D[i],-y)
                W=np.add(W,np.dot(a*e.transpose(),X[i]))
                b=np.add(b,a*e)
                E=E+np.dot(e,e.transpose())
                Err[t]=E
            t=t+1
        return W, b, Err