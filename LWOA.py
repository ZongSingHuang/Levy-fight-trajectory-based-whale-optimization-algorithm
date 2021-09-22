# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:59:58 2020

@author: ZongSing_NB

https://doi.org/10.1109/ACCESS.2017.2695498
"""

import math
import numpy as np
import matplotlib.pyplot as plt

class LWOA():
    def __init__(self, fitness, D=30, P=20, G=500, ub=1, lb=0,
                 b=1, a_max=2, a_min=0, a2_max=-1, a2_min=-2, l_max=1, l_min=-1):
        self.fitness = fitness
        self.D = D
        self.P = P
        self.G = G
        self.ub = ub*np.ones([self.P, self.D])
        self.lb = lb*np.ones([self.P, self.D])
        self.a_max = a_max
        self.a_min = a_min
        self.a2_max = a2_max
        self.a2_min = a2_min
        self.l_max = l_max
        self.l_min = l_min
        self.b = b
        
        self.gbest_X = np.zeros([self.D])
        self.gbest_F = np.inf
        self.loss_curve = np.zeros(self.G)
        
        
    def opt(self):
        # 初始化
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        
        # 迭代
        for g in range(self.G):
            # 適應值計算
            F = self.fitness(self.X)
            
            # 更新最佳解
            if np.min(F) < self.gbest_F:
                idx = F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = F.min()
            
            # 收斂曲線
            self.loss_curve[g] = self.gbest_F
            
            # 更新
            a = self.a_max - (self.a_max-self.a_min)*(g/self.G)
            a2 = self.a2_max - (self.a2_max-self.a2_min)*(g/self.G)
            
            for i in range(self.P):
                p = np.random.uniform()
                r1 = np.random.uniform()
                r2 = np.random.uniform()
                r3 = np.random.uniform()
                A = 2*a*r1 - a #(2.3)
                C = 2*r2 # (2.4)
                l = (a2-1)*r3 + 1 # (???)
                
                if p>0.5:
                    D = np.abs(self.gbest_X - self.X[i, :])
                    self.X[i, :] = D*np.exp(self.b*l)*np.cos(2*np.pi*l)+self.gbest_X # (6)
                else:
                    if np.abs(A)<1:
                        D = np.abs(C*self.gbest_X - self.X[i, :])
                        self.X[i, :] = self.gbest_X - A*D # (2)
                    else:
                        X_rand = self.X[np.random.randint(low=0, high=self.P, size=self.D), :]
                        X_rand = np.diag(X_rand).copy()
                        D = np.abs(C*X_rand - self.X[i, :])
                        self.X[i, :] = X_rand - A*D # (9)
            
            for i in range(self.P):
                u = np.random.uniform()
                r4 = np.random.uniform()
                self.X[i, :] = self.X[i, :] +u*np.sign(r4-0.5)*self.Levyflight()
            
            # 邊界處理
            self.X = np.clip(self.X, self.lb, self.ub)
        
    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.gBest_curve[-1], 3))+']')
        plt.plot(self.gBest_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()
        
    def Levyflight(self):
        beta = 1.5
        f1 = math.gamma(1+beta)
        f2 = beta * math.gamma(1+beta) / 2
        f3 = np.sin(np.pi*beta/2)
        f4 = 2**( (beta-1)/2 )
        sigma_u = ( f1/f2 * f3/f4 ) ** (2/beta)
        sigma_v = 1.0 # (12)
        
        u = np.random.normal(0, sigma_u)
        v = np.random.normal(0, sigma_v)
        s = u / ( np.abs(v)**(1/beta) )
        
        return s
            