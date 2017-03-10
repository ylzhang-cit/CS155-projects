import numpy as np

def grad_U(Yij, Ui, Vj, reg):
    return reg * Ui - (Yij - np.dot(Ui, Vj)) * Vj   

def grad_V(Yij, Ui, Vj, reg):  
    return reg * Vj - (Yij - np.dot(Ui, Vj)) * Ui

def get_err(U, V, Y, reg=0.0):
    '''
    This fuction calculates the regularized error.
    Input:
        U:       K x M numpy array
        V:       K x N numpy array
        reg:     the coefficient of regularization term.
        Y:       a numpy array of is, js, y_ijs.
    '''
    err = 0.0
    for (i, j, Yij) in Y:
        err += 0.5 * (Yij - np.dot(U[:,i-1], V[:,j-1])) ** 2
    err /= len(Y)
    if reg != 0:
        err += 0.5 * reg * (np.linalg.norm(U, 'fro') ** 2 + np.linalg.norm(V, 'fro') ** 2)
    return err

def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    U = np.random.random((K, M)) - 0.5
    V = np.random.random((K, N)) - 0.5
    size = Y.shape[0]
    delta = None
    print("training reg = %s, k = %s, M = %s, N = %s" % (reg, K, M, N))
    indices = np.arange(size)    
    for epoch in range(max_epochs):
        # Run an epoch of SGD
        before_E_in = get_err(U, V, Y, reg)
        if epoch == 0:
            print("Epoch %3d, E_in (MSE): %s" % (epoch, before_E_in))
        np.random.shuffle(indices)
        for ind in indices:
            (i, j, Yij) = Y[ind]
            # Update U[i], V[j]
            U[:,i-1] -= eta * grad_U(Yij, U[:,i-1], V[:,j-1], reg)
            V[:,j-1] -= eta * grad_V(Yij, U[:,i-1], V[:,j-1], reg)
        # At end of epoch, print E_in
        E_in = get_err(U, V, Y, reg)
        print("Epoch %3d, E_in (MSE): %s" % (epoch + 1, E_in))
        # Compute change in E_in for first epoch
        if epoch == 0:
            delta = before_E_in - E_in
        # If E_in doesn't decrease by some fraction <eps>
        # of the initial decrease in E_in, stop early            
        elif before_E_in - E_in < eps * delta:
            break
    print('\n\n')
    return (U, V, get_err(U, V, Y))