/*
 * Copyright 2015 Christoph Jud (christoph.jud@unibas.ch)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <memory>
#include <utility>
#include <vector>

#include "GaussianProcess.h"
#include "KernelFactory.h"
#include "Kernel.h"
#include "DataParser.h"
#include "AutoRegression.h"
#include "PCA.h"

#include "itkUtils.h"
#include "MatrixIO.h"
#include "boost/filesystem.hpp"

typedef gpr::GaussianProcess<double>            GaussianProcessType;
typedef std::shared_ptr<GaussianProcessType>    GaussianProcessTypePointer;
//typedef GaussianProcessType::Pointer          GaussianProcessTypePointer;
typedef GaussianProcessType::VectorType         VectorType;
typedef GaussianProcessType::MatrixType         MatrixType;

typedef gpr::Kernel<double>                     KernelType;
typedef std::shared_ptr<KernelType>             KernelTypePointer;

typedef gpr::KernelFactory<double>              KernelFactoryType;

typedef std::vector<VectorType>                 DataVectorType;
typedef std::pair<VectorType, VectorType>       TrainingPairType;
typedef std::vector<TrainingPairType>           TrainingPairVectorType;

// ITK typedefs
typedef itk::Image<unsigned char, IMAGE_DIMENSIONS>             ImageType;
typedef itk::Image<unsigned char, IMAGE_DIMENSIONS+1>           ImageSeriesType;
typedef itk::Image<itk::Vector<double, TRANSFORM_DIMENSIONS>, TRANSFORM_DIMENSIONS>     DisplacementType;
typedef itk::Image<itk::Vector<double, TRANSFORM_DIMENSIONS>, TRANSFORM_DIMENSIONS+1>   DisplacementSeriesType;

typedef DataParser<double, ImageType, DisplacementType>         DataParserType;
typedef std::shared_ptr<DataParserType>                         DataParserTypePointer;
typedef AutoRegression<double>                  AutoRegressionType;
typedef std::shared_ptr<AutoRegressionType>     AutoRegressionTypePointer;
typedef PCA<double>                  PcaType;

// parsing data
TrainingPairVectorType GetTrainingData(const std::string& filename){
    TrainingPairVectorType train_pairs;

    bool parse = true;

    unsigned input_dimension = 0;
    unsigned output_dimension = 0;

    std::ifstream infile;
    try{
        infile.open(filename);
    }
    catch(...){
        throw std::string("GetTrainingData: could not read input file");
    }

    std::string line;

    // read header
    if(std::getline(infile, line)){
        std::istringstream iss(line);

        if (!(iss >> input_dimension >> output_dimension)) { throw std::string("GetTrainingData: could not read header"); } // error
    }
    else{
        throw std::string("GetTrainingData: could not read header");
    }

    // read rest of the data
    while (std::getline(infile, line))
    {
        VectorType v_input = VectorType::Zero(input_dimension);
        VectorType v_output = VectorType::Zero(output_dimension);

        std::istringstream iss(line);
        for(unsigned i=0; i<input_dimension; i++){
            double number;
            if (!(iss >> number)) { parse=false; break; }
            v_input[i] = number;
        }

        for(unsigned i=0; i<output_dimension; i++){
            double number;
            if (!(iss >> number)) { parse=false; break; }
            v_output[i] = number;
        }

        train_pairs.push_back(std::make_pair(v_input, v_output));
    }
    if(!parse) throw std::string("GetTrainingData: error in parsing data.");

    return train_pairs;

}

// new data parser
TrainingPairVectorType GetTrainingDataITK(const std::string& input, const std::string& output)
{
    TrainingPairVectorType train_pairs;

    boost::filesystem::path input_path(input);
    boost::filesystem::path output_path(output);
    int input_file_count = static_cast<int>(std::distance(boost::filesystem::directory_iterator(input_path),
                                                          boost::filesystem::directory_iterator()));
    int output_file_count = static_cast<int>(std::distance(boost::filesystem::directory_iterator(output_path),
                                                           boost::filesystem::directory_iterator()));

    //std::cout << std::endl;
    //std::cout << input_path << std::endl;
    //std::cout << input_file_count << std::endl;
    //std::cout << output_file_count << std::endl;

    assert(input_file_count == output_file_count);
    int file_count = static_cast<int>(input_file_count);

    // Loop over all files
    for(unsigned int itr_file = 0; itr_file < file_count; ++itr_file)
    {
        char fname[20];
        sprintf(fname, "%05d", itr_file);

        // Read data
        ImageType::Pointer input_image = ReadImage<ImageType>(input + "/" + fname + ".png");
        DisplacementType::Pointer output_image = ReadImage<DisplacementType>(output + "/" + fname + ".vtk");

        // Define data dimensions
        typename ImageType::SizeType input_size = input_image->GetLargestPossibleRegion().GetSize();
        typename DisplacementType::SizeType output_size = output_image->GetLargestPossibleRegion().GetSize();

        static unsigned int input_dim = input_size.GetSizeDimension();
        static unsigned int output_dim = output_size.GetSizeDimension();

        unsigned int input_vector_size = 1;
        unsigned int output_vector_size = TRANSFORM_DIMENSIONS;

        for(int itr_dim = 0; itr_dim < input_dim; ++itr_dim)
        {
            input_vector_size *= input_size[itr_dim];
        }

        for(int itr_dim = 0; itr_dim < output_dim; ++itr_dim)
        {
            output_vector_size *= output_size[itr_dim];
        }

//        std::cout << input_dim << std::endl;
//        std::cout << output_dim << std::endl;
//        std::cout << input_size << "\t" << input_vector_size << std::endl;
//        std::cout << output_size << "\t" << output_vector_size << std::endl;

        // Fill Eigen vector with data
        itk::ImageRegionConstIterator<ImageType> input_iterator(input_image, input_image->GetRequestedRegion());
        itk::ImageRegionConstIterator<DisplacementType> output_iterator(output_image, output_image->GetRequestedRegion());

        VectorType v_input = VectorType::Zero(input_vector_size);
        VectorType v_output = VectorType::Zero(output_vector_size);

        // input vector
        input_iterator.GoToBegin();
        unsigned long counter = 0;
        while(!input_iterator.IsAtEnd())
        {
            double number = input_iterator.Get();
            v_input[counter] = number;
            ++counter;
            ++input_iterator;
        }

        // output vector
        output_iterator.GoToBegin();
        counter = 0;
        while(!output_iterator.IsAtEnd())
        {
            auto pixel = output_iterator.Get();
            for (int itr_df = 0; itr_df < TRANSFORM_DIMENSIONS; ++itr_df)
            {
                v_output[counter] = pixel[itr_df];
                ++counter;
            }
            ++output_iterator;
        }

        train_pairs.push_back(std::make_pair(v_input, v_output));

    } // end for all files

    return train_pairs;
}


int main (int argc, char *argv[]){
    std::cout << "Gaussian process training app:" << std::endl;

    if(argc!=10 && argc!=11){
        std::cout << "Usage: " << argv[0] << " input_folder output_folder kernel_string data_noise output_gp n_inputModes n_outputModes startTrainImg n_TrainImgs [use_precomputed]" << std::endl;

        //        std::cout << "Usage: " << argv[0] << " data.csv kernel_string data_noise output_gp" << std::endl;

        //        std::cout << std::endl << "Example of a kernel string: GaussianKernel(2.3, 1.0,)" << std::endl;
        //        std::cout << "Example of an input file:" << std::endl;
        //        std::cout << "4 2" << std::endl;
        //        std::cout << "x0 x1 x2 x3 y0 y1" << std::endl;
        //        std::cout << " .... " << std::endl;
        return -1;
    }

    unsigned int itr_argv = 0;
    std::string input_folder = argv[++itr_argv];
    std::string output_folder = argv[++itr_argv];
    std::string kernel_string = argv[++itr_argv];
    double gp_sigma;
    std::stringstream ss; ss << argv[++itr_argv]; ss >> gp_sigma;
    std::string output_prefix = argv[++itr_argv];

    int n_inputModes;
    std::stringstream ss_nIn; ss_nIn << argv[++itr_argv]; ss_nIn >> n_inputModes;
    int n_outputModes;
    std::stringstream ss_nOut; ss_nOut << argv[++itr_argv]; ss_nOut >> n_outputModes;
    int ind_startTrainImg;
    std::stringstream ss_startTrain; ss_startTrain << argv[++itr_argv]; ss_startTrain >> ind_startTrainImg;
    int n_trainImages;
    std::stringstream ss_nTrain; ss_nTrain << argv[++itr_argv]; ss_nTrain >> n_trainImages;
    bool use_precomputed = false;
    if(argc==11)
    {
        use_precomputed = true;
    }


    //    std::string data_filename = argv[1];
    //    std::string kernel_string = argv[2];
    //    double gp_sigma;
    //    std::stringstream ss; ss << argv[3]; ss >> gp_sigma;
    //    std::string output_prefix = argv[4];

    //    std::cout << "Configuration: " << std::endl;
    //    std::cout << " - data: " << data_filename << std::endl;
    //    std::cout << " - kernel string: " << kernel_string << std::endl;
    //    std::cout << " - data noise: " << gp_sigma << std::endl;
    //    std::cout << " - output: " << output_prefix << std::endl << std::endl;

    try{

//        AutoRegressionType ar(2,5);
//        ar.AutoRegressionTest();
//        MatrixType X = gpr::ReadMatrix<MatrixType>("/tmp/test_theat.txt");
//        std::cout << X << std::endl;
//        PcaType pca(X, 1);
//        MatrixType f = pca.GetFeatures();
//        MatrixType b = pca.GetBasis();
//        MatrixType _X = pca.GetReconstructions(f);
//        MatrixType _f = pca.ComputeFeatures(X);
//        std::cout << "PCA performed without error for small matrix" << std::endl;

        bool matrix_test = gpr::MatrixIOTest();
        if(!matrix_test){
            throw std::string("MatrixIOTest failed");
        }


        std::cout << "Initialize Gaussian process... " << std::flush;
        KernelTypePointer kernel = KernelFactoryType::GetKernel(kernel_string);
        GaussianProcessTypePointer gp(new GaussianProcessType(kernel));
        gp->SetSigma(gp_sigma);

        std::cout << "[done]" << std::endl << "Parse data and perform PCA... " << std::flush;
        //        TrainingPairVectorType train_pairs = GetTrainingData(data_filename);
        //        TrainingPairVectorType train_pairs = GetTrainingDataITK(input_filename, output_filename);
        DataParserTypePointer parser(new DataParserType(input_folder, output_folder, output_prefix, n_inputModes, n_outputModes, ind_startTrainImg, n_trainImages, use_precomputed));
        assert(parser->GetNumberOfInputFiles == parser->GetNumberOfOutputFiles);
        TrainingPairVectorType train_pairs = parser->GetTrainingData();

        std::cout << "[done]" << std::endl << "Build Gaussian process... " << std::flush;
        auto t0 = std::chrono::system_clock::now();
        for(const auto &tp : train_pairs){
            gp->AddSample(tp.first, tp.second);
        }
        std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-t0;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s" << std::flush;


        std::cout << "[done]" << std::endl << "Perform learning... " << std::flush;
        t0 = std::chrono::system_clock::now();
        gp->Initialize();
        elapsed_seconds = std::chrono::system_clock::now()-t0;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s" << std::flush;

        std::cout << "[done]" << std::endl << "Saving Gaussian process... " << std::flush;
        t0 = std::chrono::system_clock::now();
        gp->Save(output_prefix);
        elapsed_seconds = std::chrono::system_clock::now()-t0;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s" << std::flush;
        std::cout << "[done]" << std::endl;
    }
    catch(std::string& s){
        std::cout << std::endl << "Error: " << s << std::endl;
        return -1;
    }

    return 0;
}
