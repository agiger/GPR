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
#include <chrono>

#include "GaussianProcess.h"
#include "Kernel.h"

#include "itkUtils.h"
#include "boost/filesystem.hpp"

typedef gpr::GaussianProcess<double>            GaussianProcessType;
typedef std::shared_ptr<GaussianProcessType>    GaussianProcessTypePointer;
typedef GaussianProcessType::VectorType         VectorType;
typedef GaussianProcessType::MatrixType         MatrixType;

typedef gpr::Kernel<double>                     KernelType;
typedef std::shared_ptr<KernelType>             KernelTypePointer;

typedef std::vector<VectorType>                 TestVectorType;

// ITK typedefs
typedef itk::Image<unsigned char, 2> ImageType;
typedef itk::Image<double, 3> MasterImageType;
typedef itk::Image<itk::Vector<double, TRANSFORM_DIMENSIONS>, IMAGE_DIMENSIONS> DisplacementType;

// parsing data
TestVectorType GetTestData(const std::string& filename){
    TestVectorType test_vectors;

    bool parse = true;

    unsigned input_dimension = 0;

    std::ifstream infile;
    try{
        infile.open(filename);
    }
    catch(...){
        throw std::string("GetTestData: could not read input file");
    }

    std::string line;

    // read header
    if(std::getline(infile, line)){
        std::istringstream iss(line);

        if (!(iss >> input_dimension)) { throw std::string("GetTestData: could not read header"); }
    }
    else{
        throw std::string("GetTestData: could not read header");
    }

    // read rest of the data
    while (std::getline(infile, line))
    {
        VectorType v_input = VectorType::Zero(input_dimension);

        std::istringstream iss(line);
        for(unsigned i=0; i<input_dimension; i++){
            double number;
            if (!(iss >> number)) { parse=false; break; }
            v_input[i] = number;
        }

        test_vectors.push_back(v_input);
    }
    if(!parse) throw std::string("GetTestData: error in parsing data.");

    return test_vectors;

}

// new data parser
TestVectorType GetTestDataITK(const std::string& input)
{
    TestVectorType test_vectors;

    boost::filesystem::path input_path(input);
    int file_count = std::distance(boost::filesystem::directory_iterator(input_path),
                                         boost::filesystem::directory_iterator());

//    std::cout << input_path << std::endl;
//    std::cout << file_count << std::endl;

    // Loop over all files
    for(unsigned int itr_file = 0; itr_file < file_count; ++itr_file)
    {
        char fname[20];
        int n = sprintf(fname, "%05d", itr_file);

        // Read data
        ImageType::Pointer input_image = ReadImage<ImageType>(input + "/" + fname + ".png");

        // Define data dimensions
        typename ImageType::SizeType input_size = input_image->GetLargestPossibleRegion().GetSize();

        static unsigned int input_dim = input_size.GetSizeDimension();

        unsigned int input_vector_size = 1;

        for(int itr_dim = 0; itr_dim < input_dim; ++itr_dim)
        {
            input_vector_size *= input_size[itr_dim];
        }

//        std::cout << input_dim << std::endl;
//        std::cout << input_size << "\t" << input_vector_size << std::endl;

        // Fill Eigen vector with data
        itk::ImageRegionConstIterator<ImageType> input_iterator(input_image, input_image->GetRequestedRegion());

        VectorType v_input = VectorType::Zero(input_vector_size);

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

        test_vectors.push_back(v_input);

    } // end for all files

    return test_vectors;
}


void SavePrediction(const TestVectorType& vectors, const std::string& output_dir,
                    const std::string& filename)
//                    typename DisplacementType::SizeType size)
{
    MasterImageType::Pointer reference = ReadImage<MasterImageType>(filename);
    MasterImageType::SizeType size = reference->GetLargestPossibleRegion().GetSize();
    MasterImageType::SpacingType spacing = reference->GetSpacing();
    MasterImageType::DirectionType direction = reference->GetDirection();
    MasterImageType::PointType origin = reference->GetOrigin();

    unsigned int counter_file = 0;
    for(const auto v : vectors)
    {
        typename DisplacementType::Pointer output_df = CreateDisplacement<DisplacementType>(size);
        output_df->SetSpacing(spacing);
        output_df->SetOrigin(origin);
        output_df->SetDirection(direction);
        itk::ImageRegionIterator<DisplacementType> output_iterator(output_df, output_df->GetRequestedRegion());

        unsigned long counter_v = 0;
        while(!output_iterator.IsAtEnd())
        {
            typename DisplacementType::PixelType df;
            for (int itr_dim = 0; itr_dim < TRANSFORM_DIMENSIONS; ++itr_dim)
            {
                df[itr_dim] = v[counter_v];
                ++counter_v;
            }

            output_iterator.Set(df);
            ++output_iterator;
        }

        // Write predicted df and warped reference
        char filename_df[20];
        int n = sprintf(filename_df, "dfPred%05d.vtk", counter_file);
        std::string output_path = output_dir + "/" + filename_df;
        WriteImage<DisplacementType>(output_df, output_path);

        char filename_warped[20];
        n = sprintf(filename_warped, "warpedImg%05d.vtk", counter_file);
        output_path = output_dir + "/" + filename_warped;
        MasterImageType::Pointer warped_image = WarpImage<MasterImageType, DisplacementType>(reference, output_df);
        WriteImage<MasterImageType>(warped_image, output_path);

        ++counter_file;
    }
}

int main (int argc, char *argv[]){
    std::cout << "Gaussian process prediction app:" << std::endl;

    if(argc!=5){
        //        std::cout << "Usage: " << argv[0] << " gp_prefix input.csv output.csv" << std::endl;
        std::cout << "Usage: " << argv[0] << " gp_prefix input output reference" << std::endl;
        return -1;
    }

    int itr_argv = 0;
    std::string gp_prefix = argv[++itr_argv];
    std::string input_dir= argv[++itr_argv];
    std::string output_dir = argv[++itr_argv];
    std::string reference = argv[++itr_argv];
    //typename DisplacementType::SizeType size;
    //for (unsigned int itr_size = 0; itr_size < IMAGE_DIMENSIONS; ++itr_size)
    //{
    //    std::string si = argv[++itr_argv];
    //    size[itr_size] = std::stoi(si);
    //}

    // TODO: Define spacing for output images!!!

    try{
        typedef gpr::WhiteKernel<double>            WhiteKernelType;
        typedef std::shared_ptr<WhiteKernelType>    WhiteKernelTypePointer;
        WhiteKernelTypePointer wk(new WhiteKernelType(1)); // dummy kernel
        GaussianProcessTypePointer gp(new GaussianProcessType(wk));
        gp->Load(gp_prefix);

        //        TestVectorType test_vectors = GetTestData(input_filename);
        TestVectorType test_vectors = GetTestDataITK(input_dir);
        TestVectorType output_vectors;
        for(const auto v : test_vectors){
            auto t0 = std::chrono::system_clock::now();
            output_vectors.push_back(gp->Predict(v));
            std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-t0;
            std::cout << "GP prediction done in " << elapsed_seconds.count() << "s" << std::endl;
        }
        SavePrediction(output_vectors, output_dir, reference);
    }
    catch(std::string &s){
        std::cout << "Error: " << s << std::endl;
        return -1;
    }


    return 0;
}

