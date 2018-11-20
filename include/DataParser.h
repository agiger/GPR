/*
Output * Comment
 */

#pragma once

#include <string>
#include <iostream>
#include <algorithm>
#include <experimental/filesystem>
#include <chrono>
//#include "boost/filesystem.hpp"

#include <Eigen/SVD>
#include <Eigen/Dense>
#include "GaussianProcess.h"

#include "itkUtils.h"

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

    typedef itk::Image<unsigned char, IMAGE_DIMENSIONS>             ImageType;
    typedef itk::Image<itk::Vector<double, TRANSFORM_DIMENSIONS>, TRANSFORM_DIMENSIONS>     DisplacementImageType;

    typedef Eigen::JacobiSVD<MatrixType>                    JacobiSVDType;
    typedef Eigen::BDCSVD<MatrixType>                       BDCSVDType;


    DataParser(std::string input_path, std::string output_path, std::string output_prefix, int input_modes, int output_modes, bool is_training)
    {
        isTraining = is_training;
        m_inputPath = input_path;
        m_outputPath = output_path;
        m_outputPrefix = output_prefix;
        m_inputFilecount = 0;
        m_outputFilecount = 0;
        m_numberOfPrincipalModesInput = input_modes;
        m_numberOfPrincipalModesOutput= output_modes;
    }

    ~DataParser(){}

    int GetNumberOfInputFiles()
    {
        return m_inputFilecount;
    }

    int GetNumberOfOutputFiles()
    {
        return m_outputFilecount;
    }

    TrainingPairVectorType& GetTrainingData()
    {
        SetFilePaths();
        PcaFeatureExtractionForTraining();
        CreateTrainingVectorPair();
        return m_trainingPairs;
    }

    TestVectorType& GetTestData()
    {
        SetFilePaths();
        PcaFeatureExtractionForPrediction();
        CreateTestVector();
        return m_testVector;
    }

    TestVectorType GetResults(TestVectorType predicted_features)
    {
        SetFilePaths();
        m_predictedFeatures = predicted_features;
        inversePca();
        CreatePredictionVector();

        return m_predictionVector;
    }

protected:

    void PcaFeatureExtractionForTraining()
    {
        bool usePrecomputed = ( std::experimental::filesystem::exists(m_pathInputBasis) &&
                                std::experimental::filesystem::exists(m_pathOutputBasis) );
        ParseInputFiles();
        ParseOutputFiles();
        assert(m_inputFilecount == m_outputFilecount); // use try catch instead

        // Subtract Mean
        m_inputMean = m_inputMatrix.rowwise().mean();
        m_outputMean = m_outputMatrix.rowwise().mean();
        MatrixType alignedInput = m_inputMatrix.colwise() - m_inputMean;
        MatrixType alignedOutput = m_outputMatrix.colwise() - m_outputMean;

        if(!usePrecomputed)
        {
            // Computing SVD
            auto t0 = std::chrono::system_clock::now();
            BDCSVDType inputSvd(alignedInput, Eigen::ComputeThinU);
            std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-t0;
            std::cout << "inputSvd done in " << elapsed_seconds.count() << "s" << std::endl;

            t0 = std::chrono::system_clock::now();
            BDCSVDType outputSvd(alignedOutput, Eigen::ComputeThinU);
            elapsed_seconds = std::chrono::system_clock::now()-t0;
            std::cout << "outputSvd done in " << elapsed_seconds.count() << "s" << std::endl;

            // Compute Basis
            m_numberOfPrincipalModesInput = std::min(m_numberOfPrincipalModesInput, static_cast<int>(inputSvd.matrixU().cols()));
            m_numberOfPrincipalModesOutput = std::min(m_numberOfPrincipalModesOutput, static_cast<int>(outputSvd.matrixU().cols()));

            MatrixType fullInputBasis = inputSvd.matrixU();
            MatrixType fullOutputBasis = outputSvd.matrixU();

            m_inputBasis = fullInputBasis.leftCols(m_numberOfPrincipalModesInput);
            m_outputBasis = fullOutputBasis.leftCols(m_numberOfPrincipalModesOutput);
        }
        else
        {
            // Read Basis
            m_inputBasis = ReadFromCsvFile(m_pathInputBasis);
            m_outputBasis = ReadFromCsvFile(m_pathOutputBasis);
        }

        // Compute Features
        m_inputFeatures = alignedInput.transpose() * m_inputBasis;
        m_outputFeatures = alignedOutput.transpose() * m_outputBasis;
        WriteToCsvFile(m_outputPrefix + "-inputFeatures.csv", m_inputFeatures);
        WriteToCsvFile(m_outputPrefix + "-outputFeatures.csv", m_outputFeatures);

        if(!usePrecomputed)
        {
            WriteToCsvFile(m_pathInputMean, m_inputMean);
            WriteToCsvFile(m_pathOutputMean, m_outputMean);
            WriteToCsvFile(m_pathInputBasis, m_inputBasis);
            WriteToCsvFile(m_pathOutputBasis, m_outputBasis);
            SaveInputMeanAsImage();
            SaveOutputMeanAsImage();
            SaveInputBasisAsImage();
            SaveOutputBasisAsImage();
        }

        // First approach
        //VectorType ones = VectorType::Constant(m_inputFilecount, 1, 1);
        //m_inputMean = m_inputMatrix*ones/m_inputFilecount;

        //ones = VectorType::Constant(m_outputFilecount, 1, 1);
        //m_outputMean = m_outputMatrix*ones/m_outputFilecount;

        //MatrixType inputMatrix01 = m_inputMatrix;
        //MatrixType outputMatrix01 = m_outputMatrix;
        //for(unsigned int itr_col; itr_col < m_inputFilecount; ++itr_col)
        //{
        //    inputMatrix01.col(itr_col) -= m_inputMean;
        //    outputMatrix01.col(itr_col) -= m_outputMean;
        //}
        //
        //std::cout << "Input matrix diff: Min " << (inputMatrix01-inputMatrix).minCoeff() << std::endl;
        //std::cout << "Input matrix diff: Max " << (inputMatrix01-inputMatrix).maxCoeff() << std::endl;
        //std::cout << "Output matrix diff: Min " << (outputMatrix01-outputMatrix).minCoeff() << std::endl;
        //std::cout << "Output matrix diff: Max " << (outputMatrix01-outputMatrix).maxCoeff() << std::endl;
    }

    void PcaFeatureExtractionForPrediction()
    {
        // Parse input files
        ParseInputFiles();
        ParseOutputFiles();

        // Read input mean and basis
        m_inputBasis = ReadFromCsvFile(m_pathInputBasis);
        m_inputMean = ReadFromCsvFile(m_pathInputMean);

        m_outputBasis = ReadFromCsvFile(m_pathOutputBasis);
        m_outputMean = ReadFromCsvFile(m_pathOutputMean);

        // Feature extraction
        MatrixType alignedInput = m_inputMatrix.colwise() - m_inputMean;
        m_inputFeatures = alignedInput.transpose() * m_inputBasis;
        WriteToCsvFile(m_outputPrefix + "-inputFeatures_prediction.csv", m_inputFeatures);

        MatrixType alignedOutput = m_outputMatrix.colwise() - m_outputMean;
        m_outputFeatures = alignedOutput.transpose() * m_outputBasis;
        WriteToCsvFile(m_outputPrefix + "-groundtruthFeatures_prediction.csv", m_outputFeatures);
    }

    void inversePca()
    {
        // Parse output files
        MatrixType outputFeatures(m_numberOfPrincipalModesOutput, m_predictedFeatures.size());
        unsigned int itr = 0;
        for(const auto v : m_predictedFeatures)
        {
            outputFeatures.col(itr) = v;
            itr++;
        }

        // Read output mean and basis
        m_outputBasis = ReadFromCsvFile(m_pathOutputBasis);
        m_outputMean = ReadFromCsvFile(m_pathOutputMean);

        // inverse PCA
        MatrixType alignedOutput = m_outputBasis * outputFeatures;
        m_predictedOutputMatrix =  alignedOutput.colwise() + m_outputMean;
        WriteToCsvFile(m_outputPrefix + "-outputFeatures_prediction.csv", outputFeatures.transpose());
    }

    void WriteToCsvFile(std::string filename, MatrixType matrix)
    {
        std::ofstream file;
        file.open(filename.c_str(), std::ios::out | std::ios::trunc);
        for(int i=0; i<matrix.rows(); ++i)
        {
            for(int j=0; j<matrix.cols(); ++j)
            {
                std::string value = std::to_string(matrix(i,j));
                if (j == int(matrix.cols())-1)
                {
                    file << value;
                }
                else
                {
                    file << value << ',';
                }
            }
            file << '\n';
        }
        file.close();
        std::cout << filename << " has been written" << std::endl;
    }

    MatrixType ReadFromCsvFile(const std::string & path) {
        std::ifstream indata;
        std::string line;
        std::vector<double> values;
        unsigned int rows = 0;

        indata.open(path);
        while (std::getline(indata, line)) {
            std::stringstream lineStream(line);
            std::string cell;
            while (std::getline(lineStream, cell, ',')) {
                values.push_back(std::stod(cell));
            }
            ++rows;
        }

        unsigned int cols = values.size()/rows;
        return Eigen::Map<MatrixType>(values.data(), rows, cols);
    }

    void CreateTrainingVectorPair()
    {
        for(unsigned int itr_file; itr_file < m_inputFilecount; ++itr_file)
        {
            VectorType v_input = m_inputFeatures.row(itr_file);
            VectorType v_output = m_outputFeatures.row(itr_file);
            m_trainingPairs.push_back(std::make_pair(v_input, v_output));
        }
    }

    void CreateTestVector()
    {
        for(unsigned int itr_file; itr_file < m_inputFilecount; ++itr_file)
        {
            VectorType v_input = m_inputFeatures.row(itr_file);
            m_testVector.push_back(v_input);
        }
    }

    void CreatePredictionVector()
    {
        for(unsigned int itr_file; itr_file < m_predictedOutputMatrix.cols() ; ++itr_file)
        {
            VectorType v_prediction = m_predictedOutputMatrix.col(itr_file);
            m_predictionVector.push_back(v_prediction);
        }
    }

    void ParseInputFiles()
    {
        for(const auto & p : std::experimental::filesystem::directory_iterator(m_inputPath))
        {
            m_inputFiles.push_back(p.path().string());
        }
        sort(m_inputFiles.begin(), m_inputFiles.end());
        m_inputFilecount = m_inputFiles.size();

        // Identify vector size
        typename TInputType::Pointer image = ReadImage<TInputType>(m_inputFiles.front());
        typename TInputType::SizeType size = image->GetLargestPossibleRegion().GetSize();
        static unsigned int image_dim = size.GetSizeDimension();

        m_inputVectorSize = 1;
        for (unsigned itr_dim = 0; itr_dim < image_dim; ++itr_dim) {
            m_inputVectorSize *= size[itr_dim];
        }

        // Loop over all files
        unsigned int counter_file = 0;
        m_inputMatrix = MatrixType::Zero(m_inputVectorSize, m_inputFilecount);
        for(std::string & file : m_inputFiles)
        {
            // Read data
            std::cout << file << std::endl;
            typename TInputType::Pointer image = ReadImage<TInputType>(file);

            // Fill Eigen vector with data
            itk::ImageRegionConstIterator <TInputType> image_iterator(image, image->GetRequestedRegion());

            image_iterator.GoToBegin();
            unsigned long counter_pix = 0;
            while (!image_iterator.IsAtEnd())
            {
                auto pixel = image_iterator.Get();
                m_inputMatrix(counter_pix, counter_file) = (TScalarType)pixel/(TScalarType)255;
                ++counter_pix;
                ++image_iterator;
            }
            ++counter_file;
        } // end for all files

        return;
    }

    void ParseOutputFiles()
    {
        for(const auto & p : std::experimental::filesystem::directory_iterator(m_outputPath))
        {
            m_outputFiles.push_back(p.path().string());
        }
        sort(m_outputFiles.begin(), m_outputFiles.end());
        m_outputFilecount = m_outputFiles.size();

        // Identify vector size
        typename TOutputType::Pointer image = ReadImage<TOutputType>(m_outputFiles.front());
        typename TOutputType::SizeType size = image->GetLargestPossibleRegion().GetSize();
        static unsigned int image_dim = size.GetSizeDimension();

        m_outputVectorSize = TRANSFORM_DIMENSIONS;
        for (unsigned itr_dim = 0; itr_dim < image_dim; ++itr_dim) {
            m_outputVectorSize *= size[itr_dim];
        }

        // Loop over all files
        unsigned int counter_file = 0;
        m_outputMatrix = MatrixType::Zero(m_outputVectorSize, m_outputFilecount);
        for(std::string & file : m_outputFiles)
        {
            // Read data
            std::cout << file << std::endl;
            typename TOutputType::Pointer image = ReadImage<TOutputType>(file);

            // Fill Eigen vector with data
            itk::ImageRegionConstIterator <TOutputType> image_iterator(image, image->GetRequestedRegion());

            image_iterator.GoToBegin();
            unsigned long counter_pix = 0;
            while (!image_iterator.IsAtEnd())
            {
                auto pixel = image_iterator.Get();
                for (int itr_df = 0; itr_df < TRANSFORM_DIMENSIONS; ++itr_df)
                {
                    m_outputMatrix(counter_pix, counter_file) = (TScalarType)pixel[itr_df];
                    ++counter_pix;
                }
                ++image_iterator;
            }
            ++counter_file;
        } // end for all files

        return;
    }

    void SaveInputMeanAsImage()
    {
        ImageType::Pointer reference = ReadImage<ImageType>(m_inputFiles.front());
        ImageType::SizeType size = reference->GetLargestPossibleRegion().GetSize();
        ImageType::SpacingType spacing = reference->GetSpacing();
        ImageType::DirectionType direction = reference->GetDirection();
        ImageType::PointType origin = reference->GetOrigin();

        typename ImageType::Pointer image = CreateImage<ImageType>(size);
        image->SetSpacing(spacing);
        image->SetOrigin(origin);
        image->SetDirection(direction);
        itk::ImageRegionIterator<ImageType> image_iterator(image, image->GetRequestedRegion());

        unsigned long counter_v = 0;
        while(!image_iterator.IsAtEnd())
        {
            typename ImageType::PixelType value = m_inputMean[counter_v];
            image_iterator.Set(value);
            ++image_iterator;
            ++counter_v;
        }

        // Write input mean
        std::string output_path = m_outputPrefix + "-inputMean.vtk";
        WriteImage<ImageType>(image, output_path);
    }

    void SaveInputBasisAsImage()
    {
        ImageType::Pointer reference = ReadImage<ImageType>(m_inputFiles.front());
        ImageType::SizeType size = reference->GetLargestPossibleRegion().GetSize();
        ImageType::SpacingType spacing = reference->GetSpacing();
        ImageType::DirectionType direction = reference->GetDirection();
        ImageType::PointType origin = reference->GetOrigin();

        for(unsigned int itr_basis = 0; itr_basis < m_outputBasis.cols(); itr_basis++)
        {
            VectorType v_image = m_inputBasis.col(itr_basis);

            typename ImageType::Pointer image = CreateImage<ImageType>(size);
            image->SetSpacing(spacing);
            image->SetOrigin(origin);
            image->SetDirection(direction);
            itk::ImageRegionIterator<ImageType> image_iterator(image, image->GetRequestedRegion());

            unsigned long counter_v = 0;
            while(!image_iterator.IsAtEnd())
            {
                typename ImageType::PixelType value = v_image[counter_v];
                image_iterator.Set(value);
                ++image_iterator;
                ++counter_v;
            }

            // Write input basis to file
            char filename_df[20];
            int n = sprintf(filename_df, "inputBasis%03d.vtk", itr_basis);
            std::string output_path = m_outputPrefix + "-" + filename_df;
            WriteImage<ImageType>(image, output_path);
        }
    }

    void SaveOutputBasisAsImage()
    {
        DisplacementImageType::Pointer reference = ReadImage<DisplacementImageType>(m_outputFiles.front());
        DisplacementImageType::SizeType size = reference->GetLargestPossibleRegion().GetSize();
        DisplacementImageType::SpacingType spacing = reference->GetSpacing();
        DisplacementImageType::DirectionType direction = reference->GetDirection();
        DisplacementImageType::PointType origin = reference->GetOrigin();

        for(unsigned int itr_basis = 0; itr_basis < m_outputBasis.cols(); itr_basis++)
        {
            VectorType v_image = m_outputBasis.col(itr_basis);

            typename DisplacementImageType::Pointer image_df = CreateDisplacement<DisplacementImageType>(size);
            image_df->SetSpacing(spacing);
            image_df->SetOrigin(origin);
            image_df->SetDirection(direction);
            itk::ImageRegionIterator<DisplacementImageType> image_iterator(image_df, image_df->GetRequestedRegion());

            unsigned long counter_v = 0;
            while(!image_iterator.IsAtEnd())
            {
                typename DisplacementImageType::PixelType df;
                for (int itr_dim = 0; itr_dim < TRANSFORM_DIMENSIONS; ++itr_dim)
                {
                    df[itr_dim] = v_image[counter_v];
                    ++counter_v;
                }

                image_iterator.Set(df);
                ++image_iterator;
            }

            // Write output basis to file
            char filename_df[20];
            int n = sprintf(filename_df, "outputBasis%03d.vtk", itr_basis);
            std::string output_path = m_outputPrefix + "-" + filename_df;
            WriteImage<DisplacementImageType>(image_df, output_path);
        }
    }

    void SaveOutputMeanAsImage()
    {
        DisplacementImageType::Pointer reference = ReadImage<DisplacementImageType>(m_outputFiles.front());
        DisplacementImageType::SizeType size = reference->GetLargestPossibleRegion().GetSize();
        DisplacementImageType::SpacingType spacing = reference->GetSpacing();
        DisplacementImageType::DirectionType direction = reference->GetDirection();
        DisplacementImageType::PointType origin = reference->GetOrigin();

        typename DisplacementImageType::Pointer image_df = CreateDisplacement<DisplacementImageType>(size);
        image_df->SetSpacing(spacing);
        image_df->SetOrigin(origin);
        image_df->SetDirection(direction);
        itk::ImageRegionIterator<DisplacementImageType> image_iterator(image_df, image_df->GetRequestedRegion());

        unsigned long counter_v = 0;
        while(!image_iterator.IsAtEnd())
        {
            typename DisplacementImageType::PixelType df;
            for (int itr_dim = 0; itr_dim < TRANSFORM_DIMENSIONS; ++itr_dim)
            {
                df[itr_dim] = m_outputMean[counter_v];
                ++counter_v;
            }

            image_iterator.Set(df);
            ++image_iterator;
        }

        // Write output mean
        std::string output_path = m_outputPrefix + "-outputMean.vtk";
        WriteImage<DisplacementImageType>(image_df, output_path);
    }

    void SetFilePaths()
    {
        m_fnameInputBasis = "-inputBasis_" + std::to_string(m_numberOfPrincipalModesInput) + ".csv";
        m_fnameOutputBasis = "-outputBasis_" + std::to_string(m_numberOfPrincipalModesOutput) + ".csv";
        m_pathInputBasis = m_outputPrefix + m_fnameInputBasis;
        m_pathInputMean = m_outputPrefix + "-inputMean.csv";
        m_pathOutputBasis = m_outputPrefix + m_fnameOutputBasis;
        m_pathOutputMean = m_outputPrefix + "-outputMean.csv";
    }

private:
    bool isTraining;
    std::string m_inputPath;
    std::string m_outputPath;
    std::string m_outputPrefix;

    int m_inputFilecount;
    int m_outputFilecount;
    int m_inputVectorSize;
    int m_outputVectorSize;

    std::vector<std::string> m_inputFiles;
    std::vector<std::string> m_outputFiles;

    MatrixType m_inputMatrix;
    MatrixType m_outputMatrix;
    MatrixType m_predictedOutputMatrix;

    VectorType m_inputMean;
    VectorType m_outputMean;

    int m_numberOfPrincipalModesInput;
    int m_numberOfPrincipalModesOutput;

    MatrixType m_inputBasis;
    MatrixType m_outputBasis;

    MatrixType m_inputFeatures;
    MatrixType m_outputFeatures;

    TrainingPairVectorType m_trainingPairs;
    TestVectorType m_testVector;

    TestVectorType m_predictedFeatures;
    TestVectorType m_predictionVector;

    // File handling
    std::string m_fnameInputBasis;
    std::string m_fnameOutputBasis;
    std::string m_pathInputBasis;
    std::string m_pathInputMean;
    std::string m_pathOutputBasis;
    std::string m_pathOutputMean;
};

#endif // DATAPARSER_H

