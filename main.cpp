#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib> 
#include "NeuralNet.hpp"

using namespace std;

void printVector(string vectorName, vector<double> &vect)
{
	cout << vectorName << ": ";
	for ( unsigned i = 0; i < vect.size(); ++i )
	{
		cout << vect[i] << " ";
	}
	cout << endl;
}

int main(int argc, char* argv[])
{
	//Argument parsing
	if(argc != 5)
	{
		cerr << "Usage: " << argv[0] << " HiddenLayers " <<
			"HiddenNeurons " << "LearningRateEta " <<
			"MomentumConstantAlpha" << endl;
		return 1;
	}
	cout << "Welcome to AddaNN!" << endl;
	unsigned hiddenLayers = atoi(argv[1]);
	unsigned hiddenNeurons = atoi(argv[2]);
	double learningRate = atof(argv[3]);
	double momentumConstant = atof(argv[4]);
	cout << "You will be running neural nets with the following configuration:" << endl;
	cout << "Hidden layers : " << hiddenLayers << endl;
	cout << "Hidden neurons per hidden layer : " << hiddenNeurons << endl;
	cout << "Eta : " << learningRate << endl;
	cout << "Alpha : " << momentumConstant << endl;
	
	// Create a neural net
	NeuralNet net = NeuralNet( hiddenLayers, hiddenNeurons, learningRate, 
		momentumConstant );
	cout << "Life was created!" << endl;

	// Insert input values
	string userInput, token;
	vector<double> inputVector, outputVector, targetVector;
	cout << "Please insert a input vector for the net." << endl;
	getline(cin,userInput);
	istringstream inputValues(userInput);
	while (getline(inputValues, token, ' ')) {
        inputVector.push_back( ::atof(token.c_str()) );
    }
    if( inputVector.size() < 8 || inputVector.size() > 8 )
    {
		cerr << "AddaNN accepts strictly 8 input values." << endl;
		return 1;
	}
	// Insert target values
	cout << "Please insert a target vector for the net." << endl;
	getline(cin,userInput);
	istringstream targetValues(userInput);
	while (getline(targetValues, token, ' ')) {
        targetVector.push_back( ::atof(token.c_str()) );
    }
    if( targetVector.size() < 5 || targetVector.size() > 5 )
    {
		cerr << "AddaNN accepts strictly 5 target values." << endl;
		return 1;
	}
	// Train the net
	cout << "Epoch 1" << endl;
	net.feedForward(inputVector);
	net.getResult(outputVector);
	net.backPropagation(targetVector);
	
	printVector("Input",inputVector);
	printVector("Output",outputVector);
	printVector("Target",targetVector);
	cout << "Net error: " << net.getNetError() << endl;
	
	cout << "Epoch 2" << endl;
	net.feedForward(inputVector);
	net.getResult(outputVector);
	net.backPropagation(targetVector);
	
	printVector("Input",inputVector);
	printVector("Output",outputVector);
	printVector("Target",targetVector);
	cout << "Net error: " << net.getNetError() << endl;
	
	cout << "Epoch 3" << endl;
	net.feedForward(inputVector);
	net.getResult(outputVector);
	net.backPropagation(targetVector);
	
	printVector("Input",inputVector);
	printVector("Output",outputVector);
	printVector("Target",targetVector);
	cout << "Net error: " << net.getNetError() << endl;
	
	//Termina el programa
	cout << "AddaNN says Bye-bye!" << endl;
	return 0;
}
