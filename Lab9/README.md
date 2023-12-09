# LAB 09
## Problem description
Write a local-search algorithm (eg. an EA) able to solve the *Problem* instances 1, 2, 5, and 10 on a 1000-loci genomes, using a minimum number of fitness calls. That's all.

## My solution
Started from a basic approach based on EA algorithm developed during lessons by professor, I concentrated my efforts in problem instance 1. With a simple crossover and mutation I noticed very bad improvement in fitness values and with many calls (only with very large population and more than 100.000 fitness call i reach 100% fitness).
After many changes of approach, i reach a good result sorting the initial population by fitness and than:
    -if my best individual has fitness value < 0.5 -> Select the worse as first individual and do an invertion (simply mutate all loci) and than force the crossover
    -if my best individual has 0.5 < fitness value < 0.99 -> Select one random individual in top 5 individual
    -if my best individual has fitness value = 0.99 -> Select a random individual from population
I choose to do mutation rarely (only if a random value is less than MP that is a random value between 0.01 and 0.1). In the other cases I do crossover. In order to do it, I always choose the best individual as second individual.
Crossover function is influenced by fitness value: the higher it is, the more segments I take from second individual ("fitness value<0.6 -> 1 segment"; "0.6 < fitness value <0.9 -> 2 segments"; "fitness value > 0.9 -> 3 segments").
Also the size of segments is influenced by fitness value: the higher it is and the less are the size. Cut points are choose randomly.
I do this because I thought that when I reach a good result in fitness, I only want to do small changes in my individual. Moreover, if I choose only small pieces from the best individual, it's more likely that those pieces contains only the informations needed for my purpose.
Depending on initial population, with problem 1 I reach 100% fitness in 5000-10000 fitness calls; with problem instances 2 I often reach a good value but not always 100%; with 5, the fitness value is about 50/60% and 20/30% with problem intance 10.
Unfortunately I don't have enough time to do other tests in order to improve the results on problem instances 5 and 10 and also the plot code should be reviewed. I'll try to do it in future, sorry for that.