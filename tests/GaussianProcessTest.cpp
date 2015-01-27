// compilation: g++ -I /home/jud/apps/Eigen_3.1.3/linux64/include/eigen3/ gp_main.cpp -o gpTest -std=c++0x

#include <iostream>
#include <memory>
#include <ctime>

#include "GaussianProcess.h"
#include "Kernel.h"


typedef GaussianKernel<double>		KernelType;
typedef std::shared_ptr<KernelType> KernelTypePointer;
typedef GaussianProcess<double> GaussianProcessType;
typedef std::shared_ptr<GaussianProcessType> GaussianProcessTypePointer;

typedef GaussianProcessType::VectorType VectorType;
typedef GaussianProcessType::MatrixType MatrixType;

void Test1(){
	/*
	 * Test 1: scalar valued GP
	 * - try to learn sinus function
	 */
    std::cout << "Test 1: sinus regression... " << std::flush;

    KernelTypePointer k(new KernelType(1.7));
	GaussianProcessTypePointer gp(new GaussianProcessType(k));
	gp->SetSigma(0);

	unsigned number_of_samples = 10;

    // add training samples
	for(unsigned i=0; i<number_of_samples; i++){
		VectorType x(1);
		x(0) = i * 2*M_PI/number_of_samples;

		VectorType y(1);
		y(0) = std::sin(x(0));
		gp->AddSample(x,y);
	}
	gp->Initialize();


    // perform prediction
	unsigned number_of_tests = 50;
    double err = 0;
	for(unsigned i=0; i<number_of_tests; i++){
		VectorType x(1);
		x(0) = i * 2*M_PI/number_of_tests;

        err += std::fabs(gp->Predict(x)(0)-std::sin(x(0)));
	}

    std::cout << " [done]. Prediction error: " << err << std::endl;
}

void Test2(){
	/*
	 * Test 2: vectorial input and vectorial output
	 * - try to learn simultainiously sin and cos
	 */
    std::cout << "Test 2: 2D regression (sin/cos)... " << std::flush;
    KernelTypePointer k(new KernelType(1.8));
	GaussianProcessTypePointer gp(new GaussianProcessType(k));
	gp->SetSigma(0);

	unsigned number_of_samples = 10;

    // add training samples
    for(unsigned i=0; i<number_of_samples; i++){
		VectorType x(2);
		x(0) = x(1) = i * 2*M_PI/number_of_samples;

		VectorType y(2);
		y(0) = std::sin(x(0));
		y(1) = std::cos(x(1));

		gp->AddSample(x,y);
	}
	gp->Initialize();


    // perform prediction
	unsigned number_of_tests = 50;
    double err = 0;
	for(unsigned i=0; i<number_of_tests; i++){
		VectorType x(2);
		x(0) = x(1) = i * 2*M_PI/number_of_tests;

        err += std::fabs(gp->Predict(x)(0)-std::sin(x(0))) +
               std::fabs(gp->Predict(x)(1)-std::cos(x(1)));
	}

    std::cout << " [done]. Prediction error: " << err << std::endl;
}

void Test3(){
	/*
	 * Test 3: performance test, just random numbers
	 */
    std::cout << "Test 3: performance test... " << std::flush;
    KernelTypePointer k(new KernelType(2));
	GaussianProcessTypePointer gp(new GaussianProcessType(k));
	gp->SetSigma(0.01);

    unsigned number_of_samples = 2500;

	for(unsigned i=0; i<number_of_samples; i++){
		VectorType x = VectorType::Random(73);
		VectorType y = VectorType::Random(73);

		gp->AddSample(x,y);
	}

	clock_t t0 = std::clock();
	gp->Initialize();
    float t_training = ((float)(clock()-t0))/CLOCKS_PER_SEC;


	unsigned number_of_tests = 50;
    t0 = std::clock();
	for(unsigned i=0; i<number_of_tests; i++){
		VectorType x = VectorType::Random(73);
		gp->Predict(x);
	}
    float t_prediction= ((float)(clock()-t0))/CLOCKS_PER_SEC;
    std::cout << " [done]. Training time: " << t_training;
    std::cout << "sec , prediction time: " << t_prediction << " sec" << std::endl;
}

void Test4(){
	/*
	 * Test 4: vectorial input and scalar output
	 * - try to learn function over some 2D landmarks
	 */
    std::cout << "Test 4: vectorial input / scalar output..." << std::endl;
    KernelTypePointer k(new KernelType(1.8));
	GaussianProcessTypePointer gp(new GaussianProcessType(k));
    gp->SetSigma(0.000);

	{
		VectorType x(2);
		x(0) = 0;
		x(1) = 0;
		VectorType y(1);
		y(0) = 10;

		gp->AddSample(x,y);
	}

	{
		VectorType x(2);
		x(0) = 5;
		x(1) = 0;
		VectorType y(1);
		y(0) = 3;

		gp->AddSample(x,y);
	}

	{
		VectorType x(2);
		x(0) = 5;
		x(1) = 8;
		VectorType y(1);
		y(0) = 3;

		gp->AddSample(x,y);
	}

	{
		VectorType x(2);
		x(0) = 3;
		x(1) = 5;
		VectorType y(1);
		y(0) = 5;

		gp->AddSample(x,y);
	}

	gp->Initialize();


	unsigned number_of_tests = 50;
    std::cout << "[";
	for(unsigned i=0; i<number_of_tests; i++){
		std::cout << "[";
		for(unsigned j=0; j<number_of_tests; j++){
			VectorType x(2);
			x(0) = double(i)/8;
			x(1) = double(j)/8;
            if(j<number_of_tests-1){
                std::cout << gp->Predict(x) << ", ";
            }
            else{
                std::cout << gp->Predict(x);
            }
		}
        if(i<number_of_tests-1){
            std::cout << "],";
        }
        else{
            std::cout << "]";
        }
	}
    std::cout << "]" << std::endl;
}

void Test5(){
	/*
	 * Test 5: scalar valued GP
	 * 	- test if derivative process of sinus is a cosinus
	 */
    std::cout << "Test 5: test derivative process of sinus... " << std::flush;
    KernelTypePointer k(new KernelType(1));
	GaussianProcessTypePointer gp(new GaussianProcessType(k));
	//gp->DebugOn();
    gp->SetSigma(0);

	unsigned number_of_samples = 20;

    // add training samples (sinus)
	for(unsigned i=0; i<number_of_samples; i++){
		VectorType x(1);
		x(0) = i * 4*M_PI/number_of_samples;

		VectorType y(1);
		y(0) = std::sin(x(0));
		gp->AddSample(x,y);
	}
	gp->Initialize();

    // compare derivative of process (should be cosine)
	unsigned number_of_tests = 50;
    double err = 0;
	for(unsigned i=0; i<number_of_tests; i++){
		VectorType x(1);
		x(0) = i * 4*M_PI/number_of_tests;
		MatrixType D;
		VectorType v = gp->PredictDerivative(x,D);
        err += std::fabs(D(0,0) - std::cos(x(0)));
	}
    std::cout << " [done]. Predictione error: " << err << std::endl;
}

void Test6(){
	/*
     * Test 6: vectorial input and vectorial output
	 * - try to learn simultainiously derivative of sin and cos and linear curve
	 */
    std::cout << "Test 6: test derivative process of vectorial input..." << std::flush;
    KernelTypePointer k(new KernelType(1.1));
	GaussianProcessTypePointer gp(new GaussianProcessType(k));
    gp->SetSigma(0.01);
	//gp->DebugOn();

	unsigned number_of_samples = 20;

	for(unsigned i=0; i<number_of_samples; i++){
		VectorType x(2);
		x(0) = x(1) = i * 4*M_PI/number_of_samples;

		VectorType y(3);
		y(0) = std::sin(x(0));
		y(1) = std::cos(x(1));
		y(2) = x(0);

		gp->AddSample(x,y);
	}
	gp->Initialize();

	unsigned number_of_tests = 50;
    double err = 0;
	for(unsigned i=0; i<number_of_tests; i++){
		VectorType x(2);
		x(0) = x(1) = i * 4*M_PI/number_of_tests;
		MatrixType D;
		VectorType v = gp->PredictDerivative(x,D);
        err += std::fabs(D(0,0) - std::cos(x(0)));
        err += std::fabs(D(1,0) - std::sin(x(1)));
        err += std::fabs(D(2,0) - 0.5);
	}
    std::cout << "[done]. Prediction error: " << err << std::endl;
}

void Test7(){
	/*
	 * Test 7: vectorial input and scalar output
	 * - try to learn derivative function over some 2D landmarks
	 */
	KernelTypePointer k(new KernelType(1));
		GaussianProcessTypePointer gp(new GaussianProcessType(k));
		gp->SetSigma(0.001);

		{
			VectorType x(2);
			x(0) = 0;
			x(1) = 0;
			VectorType y(1);
			y(0) = 10;

			gp->AddSample(x,y);
		}

		{
			VectorType x(2);
			x(0) = 5;
			x(1) = 0;
			VectorType y(1);
			y(0) = 3;

			gp->AddSample(x,y);
		}

		{
			VectorType x(2);
			x(0) = 5;
			x(1) = 8;
			VectorType y(1);
			y(0) = 3;

			gp->AddSample(x,y);
		}

		{
			VectorType x(2);
			x(0) = 3;
			x(1) = 5;
			VectorType y(1);
			y(0) = 5;

			gp->AddSample(x,y);
		}

		gp->Initialize();


		std::cout << "Predictions: " << std::endl;
		unsigned number_of_tests = 50;
		for(unsigned i=0; i<number_of_tests; i++){
			for(unsigned j=0; j<number_of_tests; j++){
				VectorType x(2);
				x(0) = double(i)/8;
				x(1) = double(j)/8;
				MatrixType D;
				VectorType v = gp->PredictDerivative(x,D);
				std::cout << D << std::endl << std::endl;
				//std::cout << v << ',';
			}
		}
}

int main (int argc, char *argv[]){

    Test1();
    Test2();
    Test3();
    Test4();
    Test5();
    Test6();
    //Test7(); // todo

	return 0;
}