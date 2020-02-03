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
#include "nlohmann/json.hpp"
using json = nlohmann::json;

#include "itkUtils.h"
#include "MatrixIO.h"

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


int main (int argc, char *argv[]) {
    std::cout << "\nGaussian process training app:" << std::endl;
    if (argc !=6 && argc !=7){
        std::cout << "Usage: " << argv[0] << " <path/to/config_model.json> <path/to/config_model.json>"
                                             " gpr_prefix input_folder output_folder"
                                             " [ar_folder]" << std::endl;
        return -1;
    }

    unsigned int itr_argv = 0;
    std::ifstream ifs_model(argv[++itr_argv]);
    json config_model = json::parse(ifs_model);
    std::ifstream ifs_learn(argv[++itr_argv]);
    json config_learn = json::parse(ifs_learn);
    std::string gpr_prefix = argv[++itr_argv];

    std::string input_folder = argv[++itr_argv];
    std::string output_folder = argv[++itr_argv];
    std::string ar_folder = "";
    if(config_model["perform_ar"].get<bool>()){
        ar_folder = argv[++itr_argv];
    }

    // TODO: check config file for exceptions
    // iterate the array
//    for (json::iterator it = config.begin(); it != config.end(); ++it) {
//        std::cout << *it << '\n';
//    }

//    auto batchSize = config["ar_batchSize"];
//    std::cout << batchSize[0] << " " << batchSize[1] << std::endl;

    // GP configuration
    std::string kernel_string = config_model["kernel_string"].get<std::string>();
    double data_noise = config_model["data_noise"].get<double>();

    std::cout << "Configuration: " << std::endl;
    std::cout << " - kernel string: " << kernel_string << std::endl;
    std::cout << " - data noise: " << data_noise << std::endl;
    std::cout << " - gpr prefix: " << gpr_prefix << std::endl;
    std::cout << " - input data: " << input_folder << std::endl;
    std::cout << " - output data: " << output_folder << std::endl;
    std::cout << " - ar data: " << ar_folder << std::endl << std::endl;

    try{
        std::cout << "Initialize Gaussian process... " << std::flush;
        KernelTypePointer kernel = KernelFactoryType::GetKernel(kernel_string);
        GaussianProcessTypePointer gp(new GaussianProcessType(kernel));
        gp->SetSigma(data_noise);

        std::cout << "[done]" << std::endl << "Parse data and perform PCA... " << std::endl;
        DataParserTypePointer parser(new DataParserType(input_folder, output_folder, ar_folder, gpr_prefix, config_model, config_learn));
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
        gp->Save(gpr_prefix);
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
