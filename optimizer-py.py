#!/usr/bin/env python3
"""Gradient descent optimizers: SGD, Momentum, Adam."""
import sys,math

class SGD:
    def __init__(self,lr=0.01):self.lr=lr
    def step(self,params,grads):return [p-self.lr*g for p,g in zip(params,grads)]

class Momentum:
    def __init__(self,lr=0.01,mu=0.9):self.lr=lr;self.mu=mu;self.v=None
    def step(self,params,grads):
        if self.v is None:self.v=[0]*len(params)
        self.v=[self.mu*v+self.lr*g for v,g in zip(self.v,grads)]
        return [p-v for p,v in zip(params,self.v)]

class Adam:
    def __init__(self,lr=0.001,b1=0.9,b2=0.999,eps=1e-8):
        self.lr=lr;self.b1=b1;self.b2=b2;self.eps=eps;self.m=None;self.v=None;self.t=0
    def step(self,params,grads):
        self.t+=1
        if self.m is None:self.m=[0]*len(params);self.v=[0]*len(params)
        self.m=[self.b1*m+(1-self.b1)*g for m,g in zip(self.m,grads)]
        self.v=[self.b2*v+(1-self.b2)*g*g for v,g in zip(self.v,grads)]
        mh=[m/(1-self.b1**self.t) for m in self.m]
        vh=[v/(1-self.b2**self.t) for v in self.v]
        return [p-self.lr*m/(math.sqrt(v)+self.eps) for p,m,v in zip(params,mh,vh)]

def rosenbrock(x,y):return(1-x)**2+100*(y-x**2)**2
def rosenbrock_grad(x,y):return[-2*(1-x)-400*x*(y-x**2),200*(y-x**2)]

def main():
    if len(sys.argv)>1 and sys.argv[1]=="--test":
        # SGD on quadratic
        opt=SGD(0.1);p=[5.0]
        for _ in range(100):p=opt.step(p,[2*p[0]])  # min x^2
        assert abs(p[0])<0.01
        # Adam on Rosenbrock
        adam=Adam(lr=0.01);p=[0.0,0.0]
        for _ in range(5000):
            g=rosenbrock_grad(p[0],p[1]);p=adam.step(p,g)
        assert abs(p[0]-1)<0.1 and abs(p[1]-1)<0.1,f"Got {p}"
        # Momentum
        mom=Momentum(0.01,0.9);p=[5.0]
        for _ in range(200):p=mom.step(p,[2*p[0]])
        assert abs(p[0])<0.1
        print("All tests passed!")
    else:
        adam=Adam(lr=0.01);p=[-1.0,1.0]
        for i in range(3000):
            g=rosenbrock_grad(p[0],p[1]);p=adam.step(p,g)
        print(f"Rosenbrock minimum ≈ ({p[0]:.4f}, {p[1]:.4f}), f={rosenbrock(p[0],p[1]):.6f}")
if __name__=="__main__":main()
