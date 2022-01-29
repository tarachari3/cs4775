dataExport.py
	command line call (OS x) : python ~/dataExport.py
	main function:

		getNeuralData() #get’s necessary data from dataset, for neural net
		#uses root directory of '/Users/tarachari/Desktop/CS/Final_Proj/cysdataset/profiles'
		#change root variable to change directory of data


neuralNet.py
	command line call (OS x): python ~/neuralNet.py

	main function:
	
		runNeuralNet() #runs neural net, get’s accuracy from predictions and saves emission probs (prints results)


HNNlog.py
	command line call (OS x): python ~/HNNlog.py

	#calls fwd,back, and viterbi functions to get predictions from HNN for cysteine bonding states
	#prints results


cysdataset:
	
	#Has Martelli study data and BLAST profiles of the protein sequences
	#Is the information passed used in dataExport.py
	#Has separate README


	
		
		
