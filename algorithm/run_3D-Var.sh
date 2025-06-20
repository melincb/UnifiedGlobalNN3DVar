#!/bin/bash
echo -e "You've entered the script for Ensemble 3D-Var."
echo -e "You should've set the following variables:"
echo -e "\tDo you want to assimilate data with Ensemble 3D-Var (i.e. to run the computation)? y/n (variable DoIPrepare)"	#DoIPrepare
echo -e "\tDo you want to plot the results? y/n (variable DoIPlot)"	#DoIPlot

DoIPrepare=$1
DoIPlot=$2

parallelProcesses=1	# Number of processes run in parallel - we never opted for parallelization, therefore this is 1

echo -e "The variables you've set are:"
echo -e "\tDoIPrepare: $DoIPrepare"
echo -e "\tDoIPlot: $DoIPlot"

echo -e "\nSleeping 5 seconds (might be a proper moment to kill the program...)"
sleep 5


AlgorithmSettings=$3
if [ ${#AlgorithmSettings} -gt 0 ]; then
	# The program will use the settings which are passed in the terminal
	echo "Using the settings passed in terminal: $AlgorithmSettings"
	sleep 3
else
	# The program will use the settings specified in the line below:
	AlgorithmSettings="--obs_qty=Z500 --obs_inc=30.0 --obs_std=10.0 --custom_addon=singobs_Ljubljana_SGD --singobs_lat=46.1 --singobs_lon=14.5 --savefig_dir=Ljubljana --plot_singles --ensemble=100 --init_lr=0.3 --preconditioned_3D_Var --obs_datetime=2020-04-15-00"
	echo "Using the settings written inside this script: $AlgorithmSettings"
	sleep 3
	
fi



prepare="--compute"
plot="--plot"

start_time=$(date '+%Y-%m-%d %H:%M:%S')
echo $start_time

case $DoIPrepare in
	[Yy] )	echo -e "\n\nMoving previous algorithm inputs (if they exist)"
		if [ -e "experiments/data/20ch/algorithm_inputs.pkl" ]; then
			mv experiments/data/20ch/algorithm_inputs.pkl experiments/data/20ch/algorithm_inputs_previous.pkl
		fi
		echo -e "\n\nPreparing data by running the following command:"
		echo -e "python prepare_or_plot.py $AlgorithmSettings $prepare"
		sleep 1
		python prepare_or_plot.py $AlgorithmSettings $prepare
		echo -e "\n\nExecuting the Ensemble 3D-Var algorithm by running the following command:"
		if [ "$parallelProcesses" -eq 1 ]; then
			echo "python algorithm-serial.py"
			sleep 3
			python algorithm-serial.py
		# Check if the variable is greater than 1
		elif [ "$parallelProcesses" -gt 1 ]; then
			echo "python algorithm-parallel.py"
			echo "Using $parallelProcesses parallel processes"
			sleep 3
			python algorithm-parallel.py
		else
		    echo "Invalid number of parallel processes: $parallelProcesses"
		    exit
		fi
		
		echo -e "\n\nComputation ended!"
		;;
	*)	;;
esac

case $DoIPlot in
	[Yy] )	echo -e "\n\nPlotting results by running the following command:"
		echo "python prepare_or_plot.py $AlgorithmSettings $plot"
		sleep 1
		python prepare_or_plot.py $AlgorithmSettings $plot
		echo -e "\n\nPlotting ended!"
		;;
	*)	;;
esac

echo "Bash script finished!"
echo "Ran from $start_time to $(date '+%Y-%m-%d %H:%M:%S')"
