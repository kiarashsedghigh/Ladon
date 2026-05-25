from prob_dist import *


if __name__ == "__main__":
    a = ProbabilityDistribution([0,-1], [1/2, 1/2])
    b = ProbabilityDistribution([0,1], [1/2, 1/2])
    c = a * b
    c = c * c

    d = sample_binomial_distribution(2)

    c.print_probabilities()
    d.print_probabilities()



