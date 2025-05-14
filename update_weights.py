"""Compute dual ascent update for a given constraint

Args: 
    lamb: lambda (Lagrange multiplier) vector
    rho: penalty weight vector
    C: constraints (values precomputed)
    k_nci: number of inequality constraints

Returns:
    lamb: updated Lagrangian multiplier vector
"""
def dual_ascent_update(lamb, rho, C, k_nci):
    lamb = lamb + rho*C

    # ineq constraints: lambda = max(0, lambda + rho*C)
    for i in range(k_nci):
        lamb[i] = max(0, lamb)

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
