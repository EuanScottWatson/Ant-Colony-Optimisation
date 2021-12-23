# Ant Colony Optimisation
This is an algorithm which models the behaviour of ants to find a solution to the Travelling Salesman Problem. Each ant will pick a next point to visit with a weighted probability based on how far away that node is and the strength of the pheremone leading there. The pheremones are layed by ants from previous generations who have already found their solution. The strength of this pheremone is dependant on how well that ant's solution was compared to all others. Read [this wikipedia](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms) page for a more detailed explanation.

# Running the Code
To run the code, simply call:
```
python simulation.py ant=<int> points=<int> alpha=<float> beta=<float> diffusion_rate=<float>
```
where:
1. `ant` = number of ants finding paths
2. `points` = number of points in the graph
3. `alpha` = alpha hyperparameter for distance factor
4. `beta` = beta hyperparameter for pheremone strength factor
5. `diffusion_rate` = rate at which the pheremones diffuse [0, 1]

However, these parameters can be left blank at which point the default values will be chosen. <br>

While running the code you can press one of three buttons:
1. `Esc`: to quit the simulation
2. `p`: to view all the pheremone trails - the more red, the more intense
3. `a`: to view all the ants paths of that generation
The blue path is always the best path found so far.

# Example
In the two graphs below you can see how the best path diminishes with time, in these scenarios the number of ants and points were 50 and 25 then 100 and 50.

![Stuff here](/graphs/graph_50_25.png)
![alt text](/graphs/graph_100_50.png)