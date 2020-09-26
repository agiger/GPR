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
#include <ctime>

#include "GaussianProcess.h"
#include "Kernel.h"

#include "DataParser.h"
#include "itkUtils.h"
#include "logUtils.h"
#include "boost/filesystem.hpp"

typedef gpr::GaussianProcess<double>            GaussianProcessType;
typedef std::shared_ptr<GaussianProcessType>    GaussianProcessTypePointer;
typedef GaussianProcessType::VectorType         VectorType;
typedef GaussianProcessType::MatrixType         MatrixType;

typedef gpr::Kernel<double>                     KernelType;
typedef std::shared_ptr<KernelType>             KernelTypePointer;

typedef std::vector<VectorType>                 TestVectorType;

// ITK typedefs
typedef itk::Image<unsigned char, 2>            ImageType;
typedef itk::Image<double, 3>                   MasterImageType;
typedef itk::Image<itk::Vector<double, TRANSFORM_DIMENSIONS>, IMAGE_DIMENSIONS>     DisplacementType;

typedef DataParser<double, ImageType, DisplacementType >           DataParserType;
typedef std::shared_ptr<DataParserType>         DataParserTypePointer;


void SavePrediction(const TestVectorType& vectors, const std::string& output_dir,
                    const std::string& filename)
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
        ++counter_file;
    }
}

void WriteVectorToFile(std::string filename, std::vector<double> vec)
{
    std::ofstream file;
    file.open(filename.c_str(), std::ios::out | std::ios::app);
    for(const auto v: vec)
    {
        file << v << ',';
    }
    file.close();
}

int main (int argc, char *argv[]){
  std::cout << "\nGaussian process prediction app:" << std::endl;
  if (argc !=8){
    std::cout << "Usage: " << argv[0] << " <path/to/config_model.json> <path/to/config_predict.json>"
      " gpr_prefix input_folder groundtruth_folder"
      " result_folder reference_file" << std::endl;
    return -1;
  }

  unsigned int itr_argv = 0;
  std::ifstream ifs_model(argv[++itr_argv]);
  json config_model = json::parse(ifs_model);
  std::ifstream ifs_predict(argv[++itr_argv]);
  json config_predict = json::parse(ifs_predict);
  std::string gpr_prefix = argv[++itr_argv];

  std::string input_folder = argv[++itr_argv];
  std::string groundtruth_folder = argv[++itr_argv];
  std::string result_folder = argv[++itr_argv];
  std::string reference = argv[++itr_argv];

  // GP configuration
  std::string kernel_string = config_model["kernel_string"].get<std::string>();
  double data_noise = config_model["data_noise"].get<double>();

  std::cout << "Configuration: " << std::endl;
  std::cout << " - kernel string: " << kernel_string << std::endl;
  std::cout << " - data noise: " << data_noise << std::endl;
  std::cout << " - gpr prefix: " << gpr_prefix << std::endl;
  std::cout << " - input data: " << input_folder << std::endl;
  std::cout << " - ground truth data: " << groundtruth_folder << std::endl;
  std::cout << " - result folder: " << result_folder << std::endl;
  std::cout << " - reference file: " << reference<< std::endl << std::endl;
  std::cout << "Use precomputed PCA paramteres: " << config_predict["use_precomputed"] << std::endl << std::endl;

  // Write GP configuration to log file
  std::string logPath = gpr_prefix + "-log_";
  writeToLogFile(logPath, "\n" + getCurrentDateTime("now"));
  writeToLogFile(logPath, "Gaussian process prediction app");
  writeToLogFile(logPath, "Configuration:");
  writeToLogFile(logPath, " - kernel string: " + kernel_string);
  writeToLogFile(logPath, " - data noise: " + std::to_string(data_noise));
  writeToLogFile(logPath, " - gpr prefix: " + gpr_prefix);
  writeToLogFile(logPath, " - input data: " + input_folder);
  writeToLogFile(logPath, " - ground truth data: " + groundtruth_folder);
  writeToLogFile(logPath, " - result folder: " + result_folder);
  writeToLogFile(logPath, " - reference file: " + reference);
  writeToLogFile(logPath, "\nUse precomputed PCA basis: " + std::to_string(config_predict["use_precomputed"].get<bool>()) + "\n");

  try{
    // Initialize GP
    std::cout << "Initialize Gaussian process... " << std::flush;
    auto t0 = std::chrono::system_clock::now();
    typedef gpr::WhiteKernel<double>            WhiteKernelType;
    typedef std::shared_ptr<WhiteKernelType>    WhiteKernelTypePointer;
    WhiteKernelTypePointer wk(new WhiteKernelType(1)); // dummy kernel
    GaussianProcessTypePointer gp(new GaussianProcessType(wk));
    gp->Load(gpr_prefix);
    std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-t0;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s [done]" << std::endl;
    writeToLogFile(logPath, "Initialize Gaussian process... elapsed time: " + std::to_string(elapsed_seconds.count()) + " [successfully completed]");

    // Parse data and extract PCA features
    std::cout << "Parse data and extract PCA features... " << std::endl;
    writeToLogFile(logPath, "Parse data and extract PCA features...");
    t0 = std::chrono::system_clock::now();
    DataParserTypePointer parser(new DataParserType(input_folder, groundtruth_folder, gpr_prefix, config_model, config_predict));
    TestVectorType test_vectors = parser->GetTestData();
    elapsed_seconds = std::chrono::system_clock::now()-t0;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s [done]" << std::endl;
    writeToLogFile(logPath, "elapsed time: " + std::to_string(elapsed_seconds.count()) + " [PCA successfully completed]");

    // Predict features
    TestVectorType predicted_features;
    std::vector<double> confidence;
    std::vector<double> time;
    std::cout << "GP prediction done in (s):" << std::endl;
    writeToLogFile(logPath, "GP prediction done in (s):");
    for(const auto v : test_vectors){
      t0 = std::chrono::system_clock::now();
      predicted_features.push_back(gp->Predict(v));
      confidence.push_back(gp->GetCredibleInterval(v));
      elapsed_seconds = std::chrono::system_clock::now()-t0;
      std::cout << elapsed_seconds.count() << std::endl;
      writeToLogFile(logPath, std::to_string(elapsed_seconds.count()));
      time.push_back(elapsed_seconds.count());
    }
    WriteVectorToFile(gpr_prefix + "-latestInferenceTime.txt", time);

    // Reconstruct output from prinicipal components
    std::cout << "Reconstruct output from prinicipal components... " << std::endl;
    writeToLogFile(logPath, "Reconstruct output from prinicipal components...");
    t0 = std::chrono::system_clock::now();
    TestVectorType output_vectors = parser->GetResults(predicted_features);
    elapsed_seconds = std::chrono::system_clock::now()-t0;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s [done]" << std::endl;
    writeToLogFile(logPath, "elapsed time: " + std::to_string(elapsed_seconds.count()) + " [successfully completed]");

    // Evaluate Computation Time
    std::vector<double> compTime = parser->GetComputationTime();
    WriteVectorToFile(gpr_prefix + "-latestCompTimePCA.txt", compTime);

    // Save results
    std::cout << "Save results... " << std::flush;
    t0 = std::chrono::system_clock::now();
    SavePrediction(output_vectors, result_folder, reference);
    elapsed_seconds = std::chrono::system_clock::now()-t0;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s [done]" << std::endl;
    writeToLogFile(logPath, "Save results... elapsed time: " + std::to_string(elapsed_seconds.count()) + " [successfully completed]");
    WriteVectorToFile(gpr_prefix + "-credibleInterval.csv", confidence);

  }
  catch(std::string &s){
    std::cout << "Error: " << s << std::endl;
    return -1;
  }


  return 0;
}

