/*
Output * Comment
 */

#pragma once

#include <string>
#include <iostream>
#include <algorithm>
#include <experimental/filesystem>
#include <chrono>
#include <math.h>
//#include "boost/filesystem.hpp"

#include <Eigen/SVD>
#include <Eigen/Dense>
#include "GaussianProcess.h"
#include "MatrixIO.h"
#include "PCA.h"

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

    typedef itk::Image<double, IMAGE_DIMENSIONS>             ImageType;
    typedef itk::Image<itk::Vector<double, TRANSFORM_DIMENSIONS>, TRANSFORM_DIMENSIONS>     DisplacementImageType;

    typedef Eigen::JacobiSVD<MatrixType>                    JacobiSVDType;
    typedef Eigen::BDCSVD<MatrixType>                       BDCSVDType;
    typedef PCA<TScalarType>                                     PcaType;

    DataParser(std::string input_path, std::string output_path, std::string output_prefix, int input_modes, int output_modes, int ind_start_train, int n_train_images, bool use_precomputed)
    {
        usePrecomputed = use_precomputed;
        useTestData = false;
        m_srcPathInput = input_path;
        m_srcPathOutput = output_path;
        m_destPrefix = output_prefix;
        m_numberOfPrincipalModesInput = input_modes;
        m_numberOfPrincipalModesOutput= output_modes;
        m_nTrainingImages = n_train_images;
        m_indStartTrain = ind_start_train;
        m_indEndTrain = m_indStartTrain + m_nTrainingImages -1;
        std::cout << "indStart: " << m_indStartTrain << std::endl;
        std::cout << "indEnd: " << m_indEndTrain << std::endl;
        std::cout << "nImgs: " << m_nTrainingImages << std::endl;
    }

    DataParser(std::string input_path, std::string output_path, std::string output_prefix, int input_modes, int output_modes, bool use_precomputed, bool use_test_data)
    {
        usePrecomputed = use_precomputed;
        useTestData = use_test_data;
        m_srcPathInput = input_path;
        m_srcPathOutput = output_path;
        m_destPrefix = output_prefix;
        m_numberOfPrincipalModesInput = input_modes;
        m_numberOfPrincipalModesOutput= output_modes;
        m_nTrainingImages = 0;
        m_indStartTrain = 0;
        m_indEndTrain = m_indStartTrain + m_nTrainingImages -1;
        std::cout << "indStart: " << m_indStartTrain << std::endl;
        std::cout << "indEnd: " << m_indEndTrain << std::endl;
        std::cout << "nImgs: " << m_nTrainingImages << std::endl;
    }

    ~DataParser(){}

    TrainingPairVectorType GetTrainingData()
    {
        SetFilePaths();
        PcaFeatureExtractionForTraining();
        CreateTrainingVectorPair();
        return m_trainingPairs;
    }

    TestVectorType GetTestData()
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
        if(!usePrecomputed)
        {
            ParseInputFiles();
            ParseOutputFiles();
            if(m_inputMatrix.cols() % m_outputMatrix.cols() != 0){
                throw std::string("Wrong number of input or output files");
            }

            // Compute PCA
            auto t0 = std::chrono::system_clock::now();
            PcaType inputPca(m_inputMatrix);
            std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-t0;
            std::cout << "inputPca done in " << elapsed_seconds.count() << "s" << std::endl;

            t0 = std::chrono::system_clock::now();
            PcaType outputPca(m_outputMatrix);
            elapsed_seconds = std::chrono::system_clock::now()-t0;
            std::cout << "outputPca done in " << elapsed_seconds.count() << "s" << std::endl;

            // Features
            m_inputFeatures = inputPca.DimensionalityReduction(m_inputMatrix, m_numberOfPrincipalModesInput);
            m_outputFeatures = outputPca.DimensionalityReduction(m_outputMatrix, m_numberOfPrincipalModesOutput);

            // Save mean and basis as images
            VectorType inputMean = inputPca.GetMean();
            MatrixType inputBasis = inputPca.GetBasis(m_numberOfPrincipalModesInput);
            WriteVectorToImage(inputMean, m_inputFiles.front(), m_destPrefixInput + "Mean.vtk");
            WriteMatrixToImageSeries(inputBasis, m_inputFiles.front(), m_destPrefixInput);

            VectorType outputMean = outputPca.GetMean();
            MatrixType outputBasis = outputPca.GetBasis(m_numberOfPrincipalModesOutput);
            WriteVectorToDisplacement(outputMean, m_outputFiles.front(), m_destPrefixOutput + "Mean.vtk");
            WriteMatrixToDisplacementSeries(outputBasis, m_outputFiles.front(), m_destPrefixOutput);

            // Explained Variance
            VectorType inputExplainedVar= inputPca.GetExplainedVariance();
            VectorType outputExplainedVar = outputPca.GetExplainedVariance();
            gpr::WriteMatrix<MatrixType>(inputExplainedVar, m_destPrefixInput + "Compactness.bin");
            gpr::WriteMatrix<MatrixType>(outputExplainedVar, m_destPrefixOutput + "Compactness.bin");

            // Write files
            inputPca.WriteMatricesToFile(m_destPrefixInput);
            outputPca.WriteMatricesToFile(m_destPrefixOutput);

            MatrixType fullInputFeatures = inputPca.DimensionalityReduction(m_inputMatrix);
            MatrixType fullOutputFeatures = outputPca.DimensionalityReduction(m_outputMatrix);
            gpr::WriteMatrix<MatrixType>(fullInputFeatures, m_pathInputFeatures);
            gpr::WriteMatrix<MatrixType>(fullOutputFeatures, m_pathOutputFeatures);

            std::cout << "inputBasis: " << inputBasis.rows() << "x" << inputBasis.cols() << std::endl;
            std::cout << "inputFeatures: " << m_inputFeatures.rows() << "x" << m_inputFeatures.cols() << std::endl;
            std::cout << "outputBasis: " << outputBasis.rows() << "x" << outputBasis.cols() << std::endl;
            std::cout << "outputFeatures: " << m_outputFeatures.rows() << "x" << m_outputFeatures.cols() << std::endl;
        }
        else
        {
            // Read Features
            MatrixType fullInputFeatures = gpr::ReadMatrix<MatrixType>(m_pathInputFeatures);
            m_inputFeatures = fullInputFeatures.topRows(m_numberOfPrincipalModesInput);

            MatrixType fullOutputFeatures = gpr::ReadMatrix<MatrixType>(m_pathOutputFeatures);
            m_outputFeatures = fullOutputFeatures.topRows(m_numberOfPrincipalModesOutput);
        }

    }

    void PcaFeatureExtractionForPrediction()
    {
        if(!usePrecomputed)
        {
            // Parse input files
            ParseInputFiles();
            PcaType inputPca(m_destPrefixInput);

            m_inputFeatures = inputPca.DimensionalityReduction(m_inputMatrix, m_numberOfPrincipalModesInput);
            MatrixType m_fullInputFeatures = inputPca.DimensionalityReduction(m_inputMatrix);
            gpr::WriteMatrix<MatrixType>(m_fullInputFeatures, m_pathInputFeatures);

            if(!useTestData)
            {
                // Parse ground truth files
                ParseOutputFiles();
                PcaType outputPca(m_destPrefixOutput);

                m_outputFeatures = outputPca.DimensionalityReduction(m_outputMatrix, m_numberOfPrincipalModesOutput);
                MatrixType m_fullOutputFeatures = outputPca.DimensionalityReduction(m_outputMatrix);
                gpr::WriteMatrix<MatrixType>(m_fullOutputFeatures, m_pathGroundTruthFeatures);
            }
        }
        else
        {
            MatrixType fullInputFeatures = gpr::ReadMatrix<MatrixType>(m_pathInputFeaturesForPrediction);
            m_inputFeatures = fullInputFeatures.topRows(m_numberOfPrincipalModesInput);
        }
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
        gpr::WriteMatrix<MatrixType>(outputFeatures, m_pathOutputFeaturesForPrediction);
        PcaType outputPca(m_destPrefixOutput);
        m_predictedOutputMatrix = outputPca.GetReconstruction(outputFeatures);
    }

    void CreateTrainingVectorPair()
    {
        for(unsigned int itr_file; itr_file < m_inputFeatures.cols(); ++itr_file)
        {
            VectorType v_input = m_inputFeatures.col(itr_file);
            VectorType v_output = m_outputFeatures.col(itr_file);
            m_trainingPairs.push_back(std::make_pair(v_input, v_output));
        }
    }

    void CreateTestVector()
    {
        for(unsigned int itr_file; itr_file < m_inputFeatures.cols(); ++itr_file)
        {
            VectorType v_input = m_inputFeatures.col(itr_file);
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
        for(const auto & p : std::experimental::filesystem::directory_iterator(m_srcPathInput))
        {
            m_inputFiles.push_back(p.path().string());
        }
        sort(m_inputFiles.begin(), m_inputFiles.end());
        int inputFilecount = m_inputFiles.size();
        std::cout << "inputFilecount: " << inputFilecount << std::endl;

        if(m_nTrainingImages != 0)
        {
            m_inputFiles.erase(m_inputFiles.begin()+m_indEndTrain+1, m_inputFiles.end());
            m_inputFiles.erase(m_inputFiles.begin(), m_inputFiles.begin()+m_indStartTrain);
        }

        inputFilecount = m_inputFiles.size();
        std::cout << "inputFilecount: " << inputFilecount << std::endl;

        // Identify vector size
        typename TInputType::Pointer image = ReadImage<TInputType>(m_inputFiles.front());
        typename TInputType::SizeType size = image->GetLargestPossibleRegion().GetSize();
        static unsigned int image_dim = size.GetSizeDimension();

        int inputVectorSize = 1;
        for (unsigned itr_dim = 0; itr_dim < image_dim; ++itr_dim) {
            inputVectorSize *= size[itr_dim];
        }

        // Loop over all files
        unsigned int counter_file = 0;
        m_inputMatrix = MatrixType::Zero(inputVectorSize, inputFilecount);
        for(std::string & file : m_inputFiles)
        {
            // Read data
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
        for(const auto & p : std::experimental::filesystem::directory_iterator(m_srcPathOutput))
        {
            m_outputFiles.push_back(p.path().string());
        }
        sort(m_outputFiles.begin(), m_outputFiles.end());
        int outputFilecount = m_outputFiles.size();
        std::cout << "outputFilecount: " << outputFilecount << std::endl;

        if(m_nTrainingImages != 0)
        {
            m_outputFiles.erase(m_outputFiles.begin()+m_indEndTrain+1, m_outputFiles.end());
            m_outputFiles.erase(m_outputFiles.begin(), m_outputFiles.begin()+m_indStartTrain);
        }

        outputFilecount = m_outputFiles.size();
        std::cout << "outputFilecount: " << outputFilecount << std::endl;

        // Identify vector size
        typename TOutputType::Pointer image = ReadImage<TOutputType>(m_outputFiles.front());
        typename TOutputType::SizeType size = image->GetLargestPossibleRegion().GetSize();
        static unsigned int image_dim = size.GetSizeDimension();

        int outputVectorSize = TRANSFORM_DIMENSIONS;
        for (unsigned itr_dim = 0; itr_dim < image_dim; ++itr_dim) {
            outputVectorSize *= size[itr_dim];
        }

        // Loop over all files
        unsigned int counter_file = 0;
        m_outputMatrix = MatrixType::Zero(outputVectorSize, outputFilecount);
        for(std::string & file : m_outputFiles)
        {
            // Read data
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

    void WriteVectorToImage(VectorType& vector, std::string fnameReference, std::string fname){
        ImageType::Pointer reference = ReadImage<ImageType>(fnameReference);
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
            typename ImageType::PixelType value = vector(counter_v);
            image_iterator.Set(value);
            ++image_iterator;
            ++counter_v;
        }
        WriteImage<ImageType>(image, fname);
    }

    void WriteMatrixToImageSeries(MatrixType& matrix, std::string fnameReference, std::string path)
    {
        for(unsigned int itr_col = 0; itr_col < matrix.cols(); itr_col++)
        {
            VectorType v_image = matrix.col(itr_col);
            char filename_df[20];
            int n = sprintf(filename_df, "Basis%03d.vtk", itr_col);
            std::string fname = path + filename_df;
            WriteVectorToImage(v_image, fnameReference, fname);
        }
    }

    void WriteVectorToDisplacement(VectorType& vector, std::string fnameReference, std::string fname)
    {
        DisplacementImageType::Pointer reference = ReadImage<DisplacementImageType>(fnameReference);
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
                df[itr_dim] = vector(counter_v);
                ++counter_v;
            }

            image_iterator.Set(df);
            ++image_iterator;
        }
        WriteImage<DisplacementImageType>(image_df, fname);
    }

    void WriteMatrixToDisplacementSeries(MatrixType& matrix, std::string fnameReference, std::string path)
    {
        for(unsigned int itr_col = 0; itr_col < matrix.cols(); itr_col++)
        {
            VectorType v_df = matrix.col(itr_col);
            char filename_df[20];
            int n = sprintf(filename_df, "Basis%03d.vtk", itr_col);
            std::string fname = path + filename_df;
            WriteVectorToDisplacement(v_df, fnameReference, fname);
        }
    }

    void SetFilePaths()
    {
        // General
        m_destPrefixInput = m_destPrefix + "-input";
        m_destPrefixOutput = m_destPrefix + "-output";

        // Training
        m_pathInputFeatures = m_destPrefixInput + "Features.bin";
        m_pathOutputFeatures = m_destPrefixOutput + "Features.bin";

        // Prediction
        m_pathInputFeaturesForPrediction = m_destPrefixInput + "Features_prediction.bin";
        m_pathOutputFeaturesForPrediction = m_destPrefixOutput + "Features_prediction.bin";
        m_pathGroundTruthFeatures = m_destPrefix + "-groundtruthFeatures_prediction.bin";
    }

private:
    bool usePrecomputed;
    bool useTestData;

    int m_nTrainingImages;
    int m_indStartTrain;
    int m_indEndTrain;

    int m_numberOfPrincipalModesInput;
    int m_numberOfPrincipalModesOutput;

    std::string m_srcPathInput;
    std::string m_srcPathOutput;
    std::vector<std::string> m_inputFiles;
    std::vector<std::string> m_outputFiles;

    MatrixType m_inputMatrix;
    MatrixType m_outputMatrix;
    MatrixType m_predictedOutputMatrix;

    MatrixType m_inputFeatures;
    MatrixType m_outputFeatures;

    TrainingPairVectorType m_trainingPairs;
    TestVectorType m_testVector;
    TestVectorType m_predictedFeatures;
    TestVectorType m_predictionVector;

    // File handling
    std::string m_destPrefix;
    std::string m_destPrefixInput;
    std::string m_destPrefixOutput;

    std::string m_pathInputFeatures;
    std::string m_pathOutputFeatures;
    std::string m_pathInputFeaturesForPrediction;
    std::string m_pathOutputFeaturesForPrediction;
    std::string m_pathGroundTruthFeatures;
};
#endif // DATAPARSER_H

