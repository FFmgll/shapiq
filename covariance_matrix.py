import numpy as np


def compute_mu_1(d):
    nom = 0
    denom = 0
    for k in range(1,d):
        nom += 1/(d*(d-k))
        denom += 1/(k*(d-k))
    return nom/denom


def compute_mu_2(d):
    nom = 0
    denom = 0
    for k in range(2,d):
        nom += (k-1)/(d-k)
    for k in range(1,d):
        denom += 1/(k*(d-k))
    return 1/(d*(d-1))*nom/denom

def compute_harmonic_number(d):
    rslt = 0
    for k in range(1,d+1):
        rslt += 1/k
    return rslt

def compute_mu_1_tilde(d):
    mu_1 = compute_mu_1(d)
    mu_2 = compute_mu_2(d)
    h = compute_harmonic_number(d-1)
    return 2*h*(mu_1+(d-2)*mu_2)/((mu_1+(d-1)*mu_2))

def compute_mu_2_tilde(d):
    mu_1 = compute_mu_1(d)
    mu_2 = compute_mu_2(d)
    h = compute_harmonic_number(d-1)
    return -2*h*mu_2/(mu_1+(d-1)*mu_2)



if __name__=="__main__":
    d=10
    mu_1 = compute_mu_1(d)
    mu_2 = compute_mu_2(d)
    mu_1_tilde = compute_mu_1_tilde(d)
    mu_2_tilde = compute_mu_2_tilde(d)
    J = np.ones((d,d))
    I = np.identity((d))
    A = mu_2*J + (mu_1-mu_2)*I
    A_inv = np.linalg.inv(A)
    A_inv_explicit = mu_2_tilde*J + (mu_1_tilde-mu_2_tilde)*I

    assert (np.round(A_inv,6)==np.round(A_inv_explicit,6)).all()