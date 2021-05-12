all:
	python3 src/main.py

clean:
	rm -f data/model_state_dict/*
	rm -f data/experiences/box/*
	rm -f data/experiences/explore/*
	echo "Time remaining\tExplore\tBox" > data/losses
	echo "Time remaining\tExplore\tBox" > data/replay_losses
	echo "Successful iterations out of total iterations" > data/results
	echo "Iteration number\tTime Taken" > data/iter_times

clean_history:
	rm data/agent_histories/*

dependencies:
	sudo apt install python3-numpy
	sudo apt install python3-pip
	sudo pip3 install scipy matplotlib networkx ansicolors
	sudo pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl

plot:
	python3 data/analysis/find_average_losses.py data/losses data/avg_losses
	gnuplot --persist data/analysis/plotting.gpi
	echo "Most recent average time: `tail data/avg_losses -n 1 | cut -f 1`"
	echo "Number of simulations: `wc -l data/avg_losses | cut -d ' ' -f 1`"

history:
	python3 data/analysis/histogram.py data/agent_histories/