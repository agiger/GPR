/*
 * Comment
 */

#pragma once

#include <string>
#include <iostream>

#include "GaussianProcess.h"

#include "itkUtils.h"
#include "boost/filesystem.hpp"

#ifndef DATAPARSER_H
#define DATAPARSER_H

template<typename TScalarType, typename TInputType, typename TOutputType>
class DataParser {
public:
    typedef gpr::GaussianProcess<TScalarType>               GaussianProcessType;
    typedef std::shared_ptr<GaussianProcessType>            GaussianProcessTypePointer;
    typedef typename GaussianProcessType::VectorType        VectorType;
    typedef typename GaussianProcessType::MatrixType        MatrixType;
    typedef typename std::shared_ptr<VectorType>            VectorTypePointer;

    typedef std::vector<VectorType>                         DataVectorType;
    typedef typename std::pair<VectorType, VectorType>      TrainingPairType;
    typedef typename std::vector<TrainingPairType>          TrainingPairVectorType;
    typedef typename std::vector<VectorType>                TestVectorType;


    DataParser(std::string input_path, std::string output_path)
    {
        m_inputPath = input_path;
        m_outputPath = output_path;
        m_inputFilecount = 0;
        m_outputFilecount = 0;
    }

    DataParser(std::string input_path)
    {
        m_inputPath = input_path;
        m_outputPath = "";
        m_inputFilecount = 0;
        m_outputFilecount = 0;
    }

    ~DataParser(){}

//    TrainingPairVectorType GetTrainingData() {};
//    TestVectorType GetTestData() {};
//    void PerformPCA() {};

    DataVectorType GetInputFiles()
    {
        boost::filesystem::path file_path(m_inputPath);
        m_inputFilecount = static_cast<int>(std::distance(boost::filesystem::directory_iterator(file_path),
                                                          boost::filesystem::directory_iterator()));

        // Loop over all files
        for(unsigned int itr_file = 0; itr_file < m_inputFilecount; ++itr_file) {
            char fname[20];
            sprintf(fname, "%05d", itr_file);

            // Read data
            typename TInputType::Pointer image = ReadImage<TInputType>(m_inputPath + "/" + fname + ".png");
            typename TInputType::SizeType size = image->GetLargestPossibleRegion().GetSize();
            static unsigned int image_dim = size.GetSizeDimension();

            int vector_size = 1;
            for (unsigned itr_dim = 0; itr_dim < image_dim; ++itr_dim) {
                vector_size *= size[itr_dim];
            }
            std::cout << size << vector_size << std::endl;

            // Fill Eigen vector with data
            itk::ImageRegionConstIterator <TInputType> image_iterator(image, image->GetRequestedRegion());
            VectorType v = VectorType::Zero(vector_size);

            image_iterator.GoToBegin();
            unsigned long counter = 0;
            while (!image_iterator.IsAtEnd())
            {
                auto pixel = image_iterator.Get();
                v[counter] = pixel;
                ++counter;
                ++image_iterator;
            }

            m_inputFiles.push_back(v);
        } // end for all files

        return m_inputFiles;
    }

    DataVectorType GetOutputFiles()
    {
        boost::filesystem::path file_path(m_outputPath);
        m_outputFilecount = static_cast<int>(std::distance(boost::filesystem::directory_iterator(file_path),
                                                           boost::filesystem::directory_iterator()));

        // Loop over all files
        for(unsigned int itr_file = 0; itr_file < m_inputFilecount; ++itr_file) {
            char fname[20];
            sprintf(fname, "%05d", itr_file);

            // Read data
            typename TOutputType::Pointer image = ReadImage<TOutputType>(m_outputPath + "/" + fname + ".vtk");
            typename TOutputType::SizeType size = image->GetLargestPossibleRegion().GetSize();
            static unsigned int image_dim = size.GetSizeDimension();

            int vector_size = TRANSFORM_DIMENSIONS;
            for (unsigned itr_dim = 0; itr_dim < image_dim; ++itr_dim) {
                vector_size *= size[itr_dim];
            }
            std::cout << size << vector_size << std::endl;

            // Fill Eigen vector with data
            itk::ImageRegionConstIterator <TOutputType> image_iterator(image, image->GetRequestedRegion());
            VectorType v = VectorType::Zero(vector_size);

            image_iterator.GoToBegin();
            unsigned long counter = 0;
            while (!image_iterator.IsAtEnd())
            {
                auto pixel = image_iterator.Get();
                for (int itr_df = 0; itr_df < TRANSFORM_DIMENSIONS; ++itr_df)
                {
                    v[counter] = pixel[itr_df];
                    ++counter;
                }
                ++image_iterator;
            }

            m_outputFiles.push_back(v);
        } // end for all files

        return m_outputFiles;
    }

    int GetNumberOfInputFiles()
    {
        return m_inputFilecount;
    }

    int GetNumberOfOutputFiles()
    {
        return m_outputFilecount;
    }

private:
    std::string m_inputPath;
    std::string m_outputPath;

    int m_inputFilecount;
    int m_outputFilecount;

    DataVectorType m_inputFiles;
    DataVectorType m_outputFiles;
};

#endif // DATAPARSER_H

