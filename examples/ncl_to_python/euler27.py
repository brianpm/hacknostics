import numpy as np

def get_val(n, a, b):
    return n*n + a*n + b


def update_primes(n, lop):
    # have to make sure list-of-primes, lop, is up to n
    if np.max(lop) < n:
        for i in range(np.max(lop), n+1):
            if (i % 2) != 0:
                if not any(np.mod(np.array(lop), i)):
                    lop.append(i)  # because it must be prime
    return lop


def is_prime(n, lop):
    # since we have an up to date list of primes
    # that includes n:
    if n in lop:
        return True
    else:
        return False


primes = [2, 3, 5, 7]

for k in range(25):
    primes = update_primes(k, primes)
    print(f"{k} prime : {is_prime(k,primes)} (length of primes is {len(primes)})")


# # possible values of b are only primes
# for bval in np.arange(2, 1001):
#
#     if is_prime(bval, np.array(primes)[np.where(np.array(primes)<bval))

