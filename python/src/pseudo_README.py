Assumptions:
    
    lifting capacity of an agent is an integer

    weight of a box is an integer

    view radius of agent = 1

    All maps in program are dictionaries of nx.Graph()

    no extremely long obstacles

Warnings:
    To prevent an agent from chasing a box in motion, all free agents first choose their action and then agents attached to boxes choose their action.

Future work:
    See if there is a performance gain by specifying (to Gaussian) whether a visited point was free or obstacle

    Agent.how_long_attached is not being used. Find a use for it or remove it.

Ideas:

References:
    Tilemap creation tool: http://riskylab.com/tilemap/
    