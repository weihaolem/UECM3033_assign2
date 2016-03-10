import numpy as np
#Your optional code here
#You can import some modules or create additional functions

def lu(A, b):
    sol=[]
    n = len(A)
    for k in range(0, n-1):
        for i in range(k+1, n):
            if A[i,k] != 0.0:
                lam = A[i,k] / A[k,k]
                A[i, k+1:n] = A[i, k+1:n] - lam * A[k, k+1:n]
                A[i, k] = lam
    for k in range(1,n):
        b[k] = b[k] - np.dot(A[k,0:k], b[0:k])
    b[n-1]=b[n-1]/A[n-1, n-1]
    for k in range(n-2, -1, -1):
        b[k] = (b[k] - np.dot(A[k,k+1:n], b[k+1:n]))/A[k,k]
    sol=b
    return list(sol)

def sor(A, b,tol):
    sol = []
    xOld = np.empty_like(b)
    error = 1e12

    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = A - L - D

    Mj = np.dot(np.linalg.inv(D), -(L+U))
    rho_Mj = max(abs(np.linalg.eigvals(Mj)))
    w = 2/(1+np.sqrt(1-rho_Mj**2))

    T = np.linalg.inv(D+w*L)
    Lw = np.dot(T, -w*U+(1-w)*D)
    c = np.dot(T, w*b)

    while error > tol:
        x = np.dot(Lw, xOld) + c
        error = np.linalg.norm(x-xOld)/np.linalg.norm(x)
        xOld = x
    sol = x
    return list(sol)

def solve(A, b):
    # Spectral Radius Theorem stated, rho(x) < 1.This must satisfy so that SOR method
    # converge for any x^(0), and the optimal relaxation parameter, w can be
    #define in 1 < w < 2.
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = A - L - D
    Mj = np.dot(np.linalg.inv(D), -(L+U))
    rho_Mj = max(abs(np.linalg.eigvals(Mj)))
    if rho_Mj < 1 :
        condition = True
    else:
        condition = False
    
    #the method to choose due to the condition
    if condition == True:
        print('Solve by sor(A,b)')
        return sor(A,b,1e-9)
        
    else:
        print('Solve by lu(A,b)')
        return lu(A,b)

if __name__ == "__main__":
    ## import checker
    ## checker.test(lu, sor, solve)

    A = np.array([[2.,1,6], [8,3,2], [1,5,1]])
    b = np.array([9., 13, 7])
    sol = solve(A,b)
    print(sol)
    
    A = np.array([[6566, -5202, -4040, -5224, 1420, 6229],
         [4104, 7449, -2518, -4588,-8841, 4040],
         [5266,-4008,6803, -4702, 1240, 5060],
         [-9306, 7213,5723, 7961, -1981,-8834],
         [-3782, 3840, 2464, -8389, 9781,-3334],
         [-6903, 5610, 4306, 5548, -1380, 3539.]])
    b = np.array([ 17603,  -63286,   56563,  -26523.5, 103396.5, -27906])
    sol = solve(A,b)
    print(sol)
