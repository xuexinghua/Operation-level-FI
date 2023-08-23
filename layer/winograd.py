import numpy as np
from sympy import symbols, Matrix, Poly, zeros, eye, Indexed, simplify, IndexedBase, init_printing, pprint, Rational
from operator import mul
from functools import reduce
import torch

def At(a,m,n):
    return Matrix(m, n, lambda i,j: a[i]**j)

def A(a,m,n):
    return At(a, m-1, n).row_insert(m-1, Matrix(1, n, lambda i,j: 1 if j==n-1 else 0))

def T(a,n):
    return Matrix(Matrix.eye(n).col_insert(n, Matrix(n, 1, lambda i,j: -a[i]**n)))

def Lx(a,n):
    x=symbols('x')
    return Matrix(n, 1, lambda i,j: Poly((reduce(mul, ((x-a[k] if k!=i else 1) for k in range(0,n)), 1)).expand(basic=True), x))

def F(a,n):
    return Matrix(n, 1, lambda i,j: reduce(mul, ((a[i]-a[k] if k!=i else 1) for k in range(0,n)), 1))

def Fdiag(a,n):
    f=F(a,n)
    return Matrix(n, n, lambda i,j: (f[i,0] if i==j else 0))

def FdiagPlus1(a,n):
    f = Fdiag(a,n-1)
    f = f.col_insert(n-1, zeros(n-1,1))
    f = f.row_insert(n-1, Matrix(1,n, lambda i,j: (1 if j==n-1 else 0)))
    return f

def L(a,n):
    lx = Lx(a,n)
    f = F(a, n)
    return Matrix(n, n, lambda i,j: lx[i, 0].nth(j)/f[i]).T

def Bt(a,n):
    return L(a,n)*T(a,n)

def B(a,n):
    return Bt(a,n-1).row_insert(n-1, Matrix(1, n, lambda i,j: 1 if j==n-1 else 0))

def cookToomFilter(a,n,r):
    alpha = n+r-1
    f = FdiagPlus1(a,alpha)
    if f[0,0] < 0:
        f[0,:] *= -1
    AT = A(a,alpha,n).T
    G = (A(a,alpha,r).T/f).T
    BT = f * B(a,alpha).T

    return (AT,G,BT)

def transfor(win_outtile):
		a = [0,1,-1,2,-2]
		r=3
		s=1
		if win_outtile > 4:
				for i in range((win_outtile-3)//2):
						a.append(Rational(1,(i+1)*2))
						a.append(-Rational(1,(i+1)*2))            
		G = cookToomFilter(a,win_outtile,3)[1]
		array_G = np.array(G).astype(np.float32)
		array_G = torch.from_numpy(array_G).cuda()   
		#print('G: ',array_G.shape)
		AT = cookToomFilter(a,win_outtile,r)[0]
		BT = cookToomFilter(a,win_outtile,r)[2]
		for i in range (win_outtile-1, -1, -1):
				if i % s != 0:
						AT.row_del(i)
		array_AT = np.array(AT).astype(np.float32)
		array_AT = torch.from_numpy(array_AT).cuda()
		#print('AT: ',array_AT.shape)   
		array_BT = np.array(BT).astype(np.float32)
		array_BT = torch.from_numpy(array_BT).cuda()
		#print('BT: ',array_BT.shape)    
		Tarray_AT = array_AT.t()
		Tarray_BT = array_BT.t()
		Tarray_G = array_G.t()
		return array_AT,array_BT,array_G,Tarray_AT,Tarray_BT,Tarray_G
   
def winograd(U,V,array_AT,Tarray_AT): 
    ele_mul=  U*V 

    output = torch.matmul(torch.matmul(array_AT, ele_mul), Tarray_AT)

    return output