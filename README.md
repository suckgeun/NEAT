* Overall Work Flow
** preparation
1. create 100 neural networks with 50 inputs and 10 outputs; no hidden layer
2. create speciation records (currently just one; initial species with no hidden layer)

** test neural networks
1. feed forward, get result

** analyze results
1. evaluate fitness of each neural networks with the results 
(use modified fitness function such that no one species overwhelm the entire population)
2. rank neural network in each group
3. drop 50% of lesser neural networks from each species. 

** get next generation
1. from each reduced species, first, mutate (check if mutation or crossover is first)
2. then crossover
3. global innovation number, connection mutation, node mutation, weight mutation 
4. E, D, how to crossover
5. inter species crossover?

** speciation
1. speciate the neural networks using the "compatibility distance" function
2. choose one genome from "previous" generation
3. compare that to current generation
4. if compatibility threshold satisfies, assign to the species
5. if no species exists, create a new species

** repeat the process
1. go back to "test neural networks" 
