Cooperative Transport Using Multi-Agent Reinforcement Learning

“To train a team of software agents to carry out transportation tasks in a randomly generated 2D grid world using reinforcement learning.”

System Requirements
	Small agents
	Cooperative
	Decentralised
	Local communication
	Limited field of view
	Anonymity
	
Literature Survey
	A significant part of the research on multi-agent learning concerns reinforcement learning techniques. Benefits and challenges of MARL were learnt from the research papers listed below:
	A Comprehensive Survey of Multi-Agent Reinforcement Learning (Buşoniu et al) Coordinating Multi-Agent Reinforcement Learning with Limited Communication (Zhang et al)
	A Survey of Multi-Agent Pathfinding Problem (Zhou et al)
	
Reinforcement Learning
	Learn from trial and error
	Decisions affect future inputs
	Learn human-like skills
	
Agent
	Actions 
		1. Move actions
		2. Box actions
	Roles
		1. Explore environment
		2. Deliver boxes
	Attributes
		1. Lifting capacity
		2. Internal map
		3. Map Gaussian
		4. Location Cache
		5. Network
		6. Percept
		
Box
	Attributes
		1. Weight
		2. Destination
	Alternating steps
		1. Voting step
		2. Group movement step
		
Conclusion
Our system reverses the traditional transportation model. Teams of agents are trained using MARL to deliver boxes in unknown environments. We show how various features affect the efficiency of the system.

References
	1. "Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto
	2. "Deep Learning" by Ian Goodfellow Yoshua Bengio, Aaron Courville
	3. "A Comprehensive Survey of Multi-Agent Reinforcement Learning" Lucian Buşoniu, Robert Babuška, Bart De Schutter
	4. "Cooperative Manipulation and Transportation with Aerial Robots" Nathan Michael, Jonathan Fink, and Vijay Kumar
	5. "Coordinating Multi-Agent Reinforcement Learning with Limited Communication" Chongjie Zhang, and Victor Lesser
	6. "A Survey of Multi-Agent Path-finding Problem" Xin Zhou
	7. "Playing Atari with Deep Reinforcement Learning'' Volodymyr Mnih, Koray Kavukcuoglu, David Silver







