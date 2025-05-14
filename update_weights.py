"""Compute dual ascent update for a given constraint

Args: 
    lamb: lambda (Lagrange multiplier)
    rho: penalty weight
    C: matrix associated with constraint (function?)
    X: states
    U: control inputs
    ineq (bool): whether given constraint is an inequality constraint
    gamma: how to update penalty weights; gamma > 1

Returns:
    lamb: updated Lagrangian multiplier
"""
def dual_ascent_update(lamb, rho, C, X, U, ineq:bool, gamma):
    if ineq:
        lamb = max(0, lamb+rho*C(X,U))
    else:
        lamb+rho*C(X,U)
    rho = gamma*rho

    return lamb


"""Compute updated penalty weights

Args: 
    rho: penalty weight
    gamma: how to update penalty weights; gamma > 1

Returns:
    rho: updated penalty weight
"""
def increasing_schedule_update(rho, gamma):
    return rho*gamma
