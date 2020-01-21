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

    typedef itk::Image<unsigned char, IMAGE_DIMENSIONS>             ImageType;
    typedef itk::Image<itk::Vector<double, TRANSFORM_DIMENSIONS>, TRANSFORM_DIMENSIONS>     DisplacementImageType;

    typedef Eigen::JacobiSVD<MatrixType>                    JacobiSVDType;
    typedef Eigen::BDCSVD<MatrixType>                       BDCSVDType;
    typedef PCA<TScalarType>                                     PcaType;

    DataParser(std::string input_path, std::string output_path, std::string output_prefix, int input_modes, int output_modes, int ind_start_train, int n_train_images, bool use_precomputed)
    {
        isTraining = true;
        usePrecomputed = use_precomputed;
        useTestData = false;
        m_inputPath = input_path;
        m_outputPath = output_path;
        m_outputPrefix = output_prefix;
        m_inputFilecount = 0;
        m_outputFilecount = 0;
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
        isTraining = false;
        usePrecomputed = use_precomputed;
        useTestData = use_test_data;
        m_inputPath = input_path;
        m_outputPath = output_path;
        m_outputPrefix = output_prefix;
        m_inputFilecount = 0;
        m_outputFilecount = 0;
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

    int GetNumberOfInputFiles()
    {
        return m_inputFilecount;
    }

    int GetNumberOfOutputFiles()
    {
        return m_outputFilecount;
    }

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
//        bool usePrecomputed = ( std::experimental::filesystem::exists(m_pathInputU) &&
//                                std::experimental::filesystem::exists(m_pathOutputU) );
//        ParseOutputFiles();
        if(!usePrecomputed)
        {
            ParseInputFiles();
            ParseOutputFiles();
            assert(m_inputFilecount == m_outputFilecount); // use try catch instead

            // Compute PCA
            auto t0 = std::chrono::system_clock::now();
            PcaType inputPca(m_inputMatrix);
            std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-t0;
            std::cout << "inputPca done in " << elapsed_seconds.count() << "s" << std::endl;

            t0 = std::chrono::system_clock::now();
            PcaType outputPca(m_outputMatrix);
            elapsed_seconds = std::chrono::system_clock::now()-t0;
            std::cout << "outputPca done in " << elapsed_seconds.count() << "s" << std::endl;

            // Features:
            std::cout << "GetFeatures:" << std::endl;
            MatrixType _inputFeatures = inputPca.DimensionalityReduction(m_inputMatrix, m_numberOfPrincipalModesInput);
            MatrixType _outputFeatures = outputPca.DimensionalityReduction(m_outputMatrix, m_numberOfPrincipalModesOutput);

            // Basis:
            std::cout << "GetBasis:" << std::endl;
            MatrixType _inputBasis = inputPca.GetBasis(m_numberOfPrincipalModesInput);
            MatrixType _outputBasis = outputPca.GetBasis(m_numberOfPrincipalModesOutput);


            // Subtract Mean
            m_inputMean = m_inputMatrix.rowwise().mean();
            m_outputMean = m_outputMatrix.rowwise().mean();
            MatrixType alignedInput = m_inputMatrix.colwise() - m_inputMean;
            MatrixType alignedOutput = m_outputMatrix.colwise() - m_outputMean;

            // Compute SVD
            t0 = std::chrono::system_clock::now();
            BDCSVDType inputSvd(alignedInput, Eigen::ComputeThinU);
            elapsed_seconds = std::chrono::system_clock::now()-t0;
            std::cout << "inputSvd done in " << elapsed_seconds.count() << "s" << std::endl;

            t0 = std::chrono::system_clock::now();
            BDCSVDType outputSvd(alignedOutput, Eigen::ComputeThinU);
            elapsed_seconds = std::chrono::system_clock::now()-t0;
            std::cout << "outputSvd done in " << elapsed_seconds.count() << "s" << std::endl;

            // Compute Basis (sigma, U)
            //m_numberOfPrincipalModesInput = std::min(m_numberOfPrincipalModesInput, static_cast<int>(inputSvd.matrixU().cols()));
            //m_numberOfPrincipalModesOutput = std::min(m_numberOfPrincipalModesOutput, static_cast<int>(outputSvd.matrixU().cols()));

            VectorType fullInputSigma = inputSvd.singularValues()/std::sqrt((TScalarType)m_inputFilecount);
            VectorType fullOutputSigma = outputSvd.singularValues()/std::sqrt((TScalarType)m_outputFilecount);

            MatrixType fullInputBasis = inputSvd.matrixU()*fullInputSigma.asDiagonal().inverse();
            MatrixType fullOutputBasis = outputSvd.matrixU()*fullOutputSigma.asDiagonal().inverse();

            MatrixType fullInputFeatures = fullInputBasis.transpose() * alignedInput;
            MatrixType fullOutputFeatures = fullOutputBasis.transpose() * alignedOutput;

            std::cout << "inputSigma: " << fullInputSigma.rows() << "x" << fullInputSigma.cols() << std::endl;
            std::cout << "inputU: " << inputSvd.matrixU().rows() << "x" << inputSvd.matrixU().cols() << std::endl;
            std::cout << "inputFeatures: " << fullInputFeatures.rows() << "x" << fullInputFeatures.cols() << std::endl;
            std::cout << "outputSigma: " << fullOutputSigma.rows() << "x" << fullOutputSigma.cols() << std::endl;
            std::cout << "outputU: " << outputSvd.matrixU().rows() << "x" << outputSvd.matrixU().cols() << std::endl;
            std::cout << "outputFeatures: " << fullOutputFeatures.rows() << "x" << fullOutputFeatures.cols() << std::endl;

            // Compute Features (coefficients c)
            m_inputBasis = fullInputBasis.leftCols(m_numberOfPrincipalModesInput);
            m_outputBasis = fullOutputBasis.leftCols(m_numberOfPrincipalModesOutput);

            m_inputFeatures = fullInputFeatures.topRows(m_numberOfPrincipalModesInput);
            m_outputFeatures = fullOutputFeatures.topRows(m_numberOfPrincipalModesOutput);

            std::cout << "inputBasis: " << m_inputBasis.rows() << "x" << m_inputBasis.cols() << std::endl;
            std::cout << "inputFeatures: " << m_inputFeatures.rows() << "x" << m_inputFeatures.cols() << std::endl;
            std::cout << "outputBasis: " << m_outputBasis.rows() << "x" << m_outputBasis.cols() << std::endl;
            std::cout << "outputFeatures: " << m_outputFeatures.rows() << "x" << m_outputFeatures.cols() << std::endl;

            // Write files
            gpr::WriteMatrix<MatrixType>(m_inputMean, m_pathInputMean);
            gpr::WriteMatrix<MatrixType>(inputSvd.matrixU(), m_pathInputU);
            gpr::WriteMatrix<MatrixType>(fullInputSigma, m_pathInputSigma);
            gpr::WriteMatrix<MatrixType>(fullInputFeatures, m_pathInputFeatures);
            gpr::WriteMatrix<MatrixType>(m_outputMean, m_pathOutputMean);
            gpr::WriteMatrix<MatrixType>(outputSvd.matrixU(), m_pathOutputU);
            gpr::WriteMatrix<MatrixType>(fullOutputSigma, m_pathOutputSigma);
            gpr::WriteMatrix<MatrixType>(fullOutputFeatures, m_pathOutputFeatures);

            SaveInputMeanAsImage();
            SaveOutputMeanAsImage();
            SaveInputBasisAsImage();
            SaveOutputBasisAsImage();

            // Compactness
            VectorType inputCumSum = VectorType::Zero(fullInputSigma.rows());
            inputCumSum(0) = fullInputSigma(0,0);
            for(int i=0; i < fullInputSigma.rows()-1; ++i)
            {
                inputCumSum(i+1) = inputCumSum(i) + fullInputSigma(i+1,i+1);
            }

            VectorType inputC = inputCumSum / inputCumSum(fullInputSigma.rows()-1);

            VectorType outputCumSum = VectorType::Zero(fullOutputSigma.rows());
            outputCumSum(0) = fullOutputSigma(0,0);
            for(int i=0; i < fullOutputSigma.rows()-1; ++i)
            {
                outputCumSum(i+1) = outputCumSum(i) + fullOutputSigma(i+1,i+1);
            }

            VectorType outputC = outputCumSum / outputCumSum(fullOutputSigma.rows()-1);

            gpr::WriteMatrix<MatrixType>(inputC, m_outputPrefix + "-inputCompactness.bin");
            gpr::WriteMatrix<MatrixType>(outputC, m_outputPrefix + "-outputCompactness.bin");


            // TESTs:
            // Features:
            std::cout << "GetFeatures:" << std::endl;
//            MatrixType _inputFeatures = inputPca.DimensionalityReduction(m_inputMatrix, m_numberOfPrincipalModesInput);
//            MatrixType _outputFeatures = outputPca.DimensionalityReduction(m_outputMatrix, m_numberOfPrincipalModesOutput);
            if(_inputFeatures.size() != m_inputFeatures.size()){
                std::cout << m_inputFeatures.rows() << "x" << m_inputFeatures.cols() << std::endl;
                std::cout << _inputFeatures.rows() << "x" << _inputFeatures.cols() << std::endl;
                throw std::string("InputFeatures do not have the same size");
            }
            for(int i=0; i<m_inputFeatures.rows(); ++i){
                for(int j=0; j<m_inputFeatures.cols(); ++j){
//                    std::cout << "i: " << i << ", j: " << j << m_inputFeatures(i,j) << " " << _inputFeatures(i,j) << std::endl;
                    if(m_inputFeatures(i,j) != _inputFeatures(i,j)) throw std::string("InputFeatures are not the same!");
                }
            }
            std::cout << "InputFeatures are correctly computed" << std::endl;

            if(_outputFeatures.size() != m_outputFeatures.size()){
                std::cout << m_outputFeatures.rows() << "x" << m_outputFeatures.cols() << std::endl;
                std::cout << _outputFeatures.rows() << "x" << _outputFeatures.cols() << std::endl;
                throw std::string("OutputFeatures do not have the same size");
            }
            for(int i=0; i<m_outputFeatures.rows(); ++i){
                for(int j=0; j<m_outputFeatures.cols(); ++j){
//                    std::cout << "i: " << i << ", j: " << m_outputFeatures(i,j) << " " << _outputFeatures(i,j) << std::endl;
                    if(m_outputFeatures(i,j) != _outputFeatures(i,j)) throw std::string("OutputFeatures are not the same!");
                }
            }
            std::cout << "OutputFeatures are correctly computed" << std::endl;

            // Basis:
            std::cout << "GetBasis:" << std::endl;
//            MatrixType _inputBasis = inputPca.GetBasis(m_numberOfPrincipalModesInput);
//            MatrixType _outputBasis = outputPca.GetBasis(m_numberOfPrincipalModesOutput);
            if(_inputBasis.size() != m_inputBasis.size()){
                std::cout << m_inputBasis.rows() << "x" << m_inputBasis.cols() << std::endl;
                std::cout << _inputBasis.rows() << "x" << _inputBasis.cols() << std::endl;
                throw std::string("InputBasis do not have the same size");
            }
            for(int i=0; i<m_inputBasis.rows(); ++i){
                for(int j=0; j<m_inputBasis.cols(); ++j){
                    if(m_inputBasis(i,j) != _inputBasis(i,j)) throw std::string("InputBasis are not the same!");
                }
            }
            std::cout << "InputBasis are correctly computed" << std::endl;

            if(_outputBasis.size() != m_outputBasis.size()){
                std::cout << m_outputBasis.rows() << "x" << m_outputBasis.cols() << std::endl;
                std::cout << _outputBasis.rows() << "x" << _outputBasis.cols() << std::endl;
                throw std::string("OutputBasis do not have the same size");
            }
            for(int i=0; i<m_outputBasis.rows(); ++i){
                for(int j=0; j<m_outputBasis.cols(); ++j){
                    if(m_outputBasis(i,j) != _outputBasis(i,j)) throw std::string("OutputBasis are not the same!");
                }
            }
            std::cout << "InputBasis are correctly computed" << std::endl;

            // Sigma:
            std::cout << "GetSigma:" << std::endl;
            MatrixType _inputSigma = inputPca.GetEigenvalues();
            MatrixType _outputSigma = outputPca.GetEigenvalues();
            if(_inputSigma.size() != fullInputSigma.size()){
                std::cout << fullInputSigma.rows() << "x" << fullInputSigma.cols() << std::endl;
                std::cout << _inputSigma.rows() << "x" << _inputSigma.cols() << std::endl;
                throw std::string("InputSigma do not have the same size");
            }
            for(int i=0; i<fullInputSigma.rows(); ++i){
                for(int j=0; j<fullInputSigma.cols(); ++j){
                    if(fullInputSigma(i,j) != _inputSigma(i,j)) throw std::string("InputSigma are not the same!");
                }
            }
            std::cout << "InputSigma are correctly computed" << std::endl;

            if(_outputSigma.size() != fullOutputSigma.size()){
                std::cout << fullOutputSigma.rows() << "x" << fullOutputSigma.cols() << std::endl;
                std::cout << _outputSigma.rows() << "x" << _outputSigma.cols() << std::endl;
                throw std::string("OutputSigma do not have the same size");
            }
            for(int i=0; i<fullOutputSigma.rows(); ++i){
                for(int j=0; j<fullOutputSigma.cols(); ++j){
                    if(fullOutputSigma(i,j) != _outputSigma(i,j)) throw std::string("OutputSigma are not the same!");
                }
            }
            std::cout << "InputSigma are correctly computed" << std::endl;

            // Mean:
            std::cout << "GetMean:" << std::endl;
            MatrixType _inputMean= inputPca.GetMean();
            MatrixType _outputMean= outputPca.GetMean();
            if(_inputMean.size() != m_inputMean.size()){
                std::cout << m_inputMean.rows() << "x" << m_inputMean.cols() << std::endl;
                std::cout << _inputMean.rows() << "x" << _inputMean.cols() << std::endl;
                throw std::string("InputMean do not have the same size");
            }
            for(int i=0; i<m_inputMean.rows(); ++i){
                for(int j=0; j<m_inputMean.cols(); ++j){
                    if(m_inputMean(i,j) != _inputMean(i,j)) throw std::string("InputMean are not the same!");
                }
            }
            std::cout << "InputMean are correctly computed" << std::endl;

            if(_outputMean.size() != m_outputMean.size()){
                std::cout << m_outputMean.rows() << "x" << m_outputMean.cols() << std::endl;
                std::cout << _outputMean.rows() << "x" << _outputMean.cols() << std::endl;
                throw std::string("OutputMean do not have the same size");
            }
            for(int i=0; i<m_outputMean.rows(); ++i){
                for(int j=0; j<m_outputMean.cols(); ++j){
                    if(m_outputMean(i,j) != _outputMean(i,j)) throw std::string("OutputMean are not the same!");
                }
            }
            std::cout << "InputMean are correctly computed" << std::endl;

            // Explained Variance:
            std::cout << "Explained Variance:" << std::endl;
            VectorType _inputExplainedVar= inputPca.GetExplainedVariance();
            VectorType _outputExplainedVar = outputPca.GetExplainedVariance();
            if(_inputExplainedVar.size() != inputC.size()){
                std::cout << inputC.rows() << "x" << inputC.cols() << std::endl;
                std::cout << _inputExplainedVar.rows() << "x" << _inputExplainedVar.cols() << std::endl;
                throw std::string("InputExplainedVariance do not have the same size");
            }
            for(int i=0; i<inputC.rows(); ++i){
                for(int j=0; j<inputC.cols(); ++j){
                    if(inputC(i,j) != _inputExplainedVar(i,j)) throw std::string("InputExplainedVar are not the same!");
                }
            }
            std::cout << "InputExplainedVar are correctly computed" << std::endl;

            if(_outputExplainedVar.size() != outputC.size()){
                std::cout << outputC.rows() << "x" << outputC.cols() << std::endl;
                std::cout << _outputExplainedVar.rows() << "x" << _outputExplainedVar.cols() << std::endl;
                throw std::string("OutputExplainedVar do not have the same size");
            }
            for(int i=0; i<outputC.rows(); ++i){
                for(int j=0; j<outputC.cols(); ++j){
                    if(outputC(i,j) != _outputExplainedVar(i,j)) throw std::string("OutputExplainedVar are not the same!");
                }
            }
            std::cout << "OutputExplainedVar are correctly computed" << std::endl;

            std::string path = m_outputPrefix + "-input";
            inputPca.WriteMatricesToFile(path);

            path = m_outputPrefix + "-output";
            outputPca.WriteMatricesToFile(path);

        }
        else
        {
            // Read Features
            MatrixType fullInputFeatures = gpr::ReadMatrix<MatrixType>(m_pathInputFeatures);
            m_inputFeatures = fullInputFeatures.topRows(m_numberOfPrincipalModesInput);
            m_inputFilecount = m_inputFeatures.cols();
            std::cout << "m_inputFilecount: " << m_inputFilecount << std::endl;

            MatrixType fullOutputFeatures = gpr::ReadMatrix<MatrixType>(m_pathOutputFeatures);
            m_outputFeatures = fullOutputFeatures.topRows(m_numberOfPrincipalModesOutput);
            m_outputFilecount = m_outputFeatures.cols();
            std::cout << "m_outputFilecount: " << m_outputFilecount << std::endl;
        }

    }

    void PcaFeatureExtractionForPrediction()
    {
        if(!usePrecomputed)
        {
            // Parse input files
            ParseInputFiles();

            // Read input mean and basis
            MatrixType inputU = gpr::ReadMatrix<MatrixType>(m_pathInputU);
            VectorType inputSigma = gpr::ReadMatrix<MatrixType>(m_pathInputSigma);
            MatrixType fullInputBasis = inputU*inputSigma.asDiagonal().inverse();
            m_inputBasis = fullInputBasis.leftCols(m_numberOfPrincipalModesInput);
            m_inputMean = gpr::ReadMatrix<MatrixType>(m_pathInputMean);

            // Feature extraction
            MatrixType alignedInput = m_inputMatrix.colwise() - m_inputMean;
            MatrixType fullInputFeatures = fullInputBasis.transpose() * alignedInput;
            m_inputFeatures = fullInputFeatures.topRows(m_numberOfPrincipalModesInput);
            gpr::WriteMatrix<MatrixType>(fullInputFeatures, m_pathInputFeaturesForPrediction);


            // TESTs:
            std::string path = m_outputPrefix + "-input";
            PcaType inputPca(path);

            // Features:
            std::cout << "GetFeatures:" << std::endl;
            MatrixType _inputFeatures = inputPca.DimensionalityReduction(m_inputMatrix, m_numberOfPrincipalModesInput);
            if(_inputFeatures.size() != m_inputFeatures.size()){
                std::cout << m_inputFeatures.rows() << "x" << m_inputFeatures.cols() << std::endl;
                std::cout << _inputFeatures.rows() << "x" << _inputFeatures.cols() << std::endl;
                throw std::string("InputFeatures do not have the same size");
            }
            for(int i=0; i<m_inputFeatures.rows(); ++i){
                for(int j=0; j<m_inputFeatures.cols(); ++j){
                    if(m_inputFeatures(i,j) != _inputFeatures(i,j)) throw std::string("InputFeatures are not the same!");
                }
            }
            std::cout << "InputFeatures are correctly computed" << std::endl;

            // Basis:
            std::cout << "GetBasis:" << std::endl;
            MatrixType _inputBasis = inputPca.GetBasis(m_numberOfPrincipalModesInput);
            if(_inputBasis.size() != m_inputBasis.size()){
                std::cout << m_inputBasis.rows() << "x" << m_inputBasis.cols() << std::endl;
                std::cout << _inputBasis.rows() << "x" << _inputBasis.cols() << std::endl;
                throw std::string("InputBasis do not have the same size");
            }
            for(int i=0; i<m_inputBasis.rows(); ++i){
                for(int j=0; j<m_inputBasis.cols(); ++j){
                    if(m_inputBasis(i,j) != _inputBasis(i,j)) throw std::string("InputBasis are not the same!");
                }
            }
            std::cout << "InputBasis are correctly computed" << std::endl;

            if(!useTestData)
            {
                // Parse ground truth files
                ParseOutputFiles();

                MatrixType outputU = gpr::ReadMatrix<MatrixType>(m_pathOutputU);
                VectorType outputSigma = gpr::ReadMatrix<MatrixType>(m_pathOutputSigma);
                MatrixType fullOutputBasis = outputU*outputSigma.asDiagonal().inverse();
                m_outputBasis = fullOutputBasis.leftCols(m_numberOfPrincipalModesOutput);
                m_outputMean = gpr::ReadMatrix<MatrixType>(m_pathOutputMean);

                MatrixType alignedOutput = m_outputMatrix.colwise() - m_outputMean;
                MatrixType fullOutputFeatures = fullOutputBasis.transpose() * alignedOutput;
                m_outputFeatures = fullOutputFeatures.topRows(m_numberOfPrincipalModesOutput);
                gpr::WriteMatrix<MatrixType>(fullOutputFeatures, m_pathGroundTruthFeatures);


                // TESTs:
                std::string path = m_outputPrefix + "-output";
                PcaType outputPca(path);

                // Features:
                std::cout << "GetFeatures:" << std::endl;
                MatrixType _outputFeatures = outputPca.DimensionalityReduction(m_outputMatrix, m_numberOfPrincipalModesOutput);
                if(_outputFeatures.size() != m_outputFeatures.size()){
                    std::cout << m_outputFeatures.rows() << "x" << m_outputFeatures.cols() << std::endl;
                    std::cout << _outputFeatures.rows() << "x" << _outputFeatures.cols() << std::endl;
                    throw std::string("OutputFeatures do not have the same size");
                }
                for(int i=0; i<m_outputFeatures.rows(); ++i){
                    for(int j=0; j<m_outputFeatures.cols(); ++j){
                        if(m_outputFeatures(i,j) != _outputFeatures(i,j)) throw std::string("OutputFeatures are not the same!");
                    }
                }
                std::cout << "OutputFeatures are correctly computed" << std::endl;

                // Basis:
                std::cout << "GetBasis:" << std::endl;
                MatrixType _outputBasis = outputPca.GetBasis(m_numberOfPrincipalModesOutput);
                if(_outputBasis.size() != m_outputBasis.size()){
                    std::cout << m_outputBasis.rows() << "x" << m_outputBasis.cols() << std::endl;
                    std::cout << _outputBasis.rows() << "x" << _outputBasis.cols() << std::endl;
                    throw std::string("OutputBasis do not have the same size");
                }
                for(int i=0; i<m_outputBasis.rows(); ++i){
                    for(int j=0; j<m_outputBasis.cols(); ++j){
                        if(m_outputBasis(i,j) != _outputBasis(i,j)) throw std::string("OutputBasis are not the same!");
                    }
                }
                std::cout << "OutputBasis are correctly computed" << std::endl;
            }
        }
        else
        {
            MatrixType fullInputFeatures = gpr::ReadMatrix<MatrixType>(m_pathInputFeaturesForPrediction);
            m_inputFeatures = fullInputFeatures.topRows(m_numberOfPrincipalModesInput);
            m_inputFilecount = m_inputFeatures.cols();
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

        // Read output mean and basis
        MatrixType outputU = gpr::ReadMatrix<MatrixType>(m_pathOutputU);
        VectorType outputSigma = gpr::ReadMatrix<MatrixType>(m_pathOutputSigma);
        MatrixType fullOutputBasis = outputU*outputSigma.asDiagonal(); // not inverse here!
        m_outputBasis = fullOutputBasis.leftCols(m_numberOfPrincipalModesOutput);
        m_outputMean = gpr::ReadMatrix<MatrixType>(m_pathOutputMean);

        // inverse PCA
        MatrixType alignedOutput = m_outputBasis * outputFeatures;
        m_predictedOutputMatrix =  alignedOutput.colwise() + m_outputMean;

        // TESTs:
        std::string path = m_outputPrefix + "-output";
        PcaType outputPca(path);

        // Features:
        std::cout << "GetFeatures:" << std::endl;
        MatrixType _outputPredictions = outputPca.GetReconstruction(outputFeatures);
        if(_outputPredictions.size() != m_predictedOutputMatrix.size()){
            std::cout << m_predictedOutputMatrix.rows() << "x" << m_predictedOutputMatrix.cols() << std::endl;
            std::cout << _outputPredictions.rows() << "x" << _outputPredictions.cols() << std::endl;
            throw std::string("PredictedOutput do not have the same size");
        }
        for(int i=0; i<m_predictedOutputMatrix.rows(); ++i){
            for(int j=0; j<m_predictedOutputMatrix.cols(); ++j){
//                std::cout << "i: " << i << ", j: " << m_predictedOutputMatrix(i,j) << " " << _outputPredictions(i,j) << std::endl;
                if(m_predictedOutputMatrix(i,j) != _outputPredictions(i,j)) throw std::string("OutputPredictions are not the same!");
            }
        }
        std::cout << "OutputPredictions are correctly computed" << std::endl;
    }

    void CreateTrainingVectorPair()
    {
        for(unsigned int itr_file; itr_file < m_inputFilecount; ++itr_file)
        {
            VectorType v_input = m_inputFeatures.col(itr_file);
            VectorType v_output = m_outputFeatures.col(itr_file);
//            VectorType v_output = m_outputMatrix.col(itr_file);
//            std::cout << v_input.size() << " " << v_output.size() << std::endl;
            m_trainingPairs.push_back(std::make_pair(v_input, v_output));
        }
    }

    void CreateTestVector()
    {
        for(unsigned int itr_file; itr_file < m_inputFilecount; ++itr_file)
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
//            VectorType v_prediction = m_predictedFeatures[itr_file];
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
        std::cout << "inputFilecount: " << m_inputFilecount << std::endl;

        if(m_nTrainingImages != 0)
        {
            m_inputFiles.erase(m_inputFiles.begin()+m_indEndTrain+1, m_inputFiles.end());
            m_inputFiles.erase(m_inputFiles.begin(), m_inputFiles.begin()+m_indStartTrain);
        }

        m_inputFilecount = m_inputFiles.size();
        std::cout << "inputFilecount: " << m_inputFilecount << std::endl;

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
//            std::cout << file << std::endl;
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
        std::cout << "outputFilecount: " << m_outputFilecount << std::endl;

        if(m_nTrainingImages != 0)
        {
            m_outputFiles.erase(m_outputFiles.begin()+m_indEndTrain+1, m_outputFiles.end());
            m_outputFiles.erase(m_outputFiles.begin(), m_outputFiles.begin()+m_indStartTrain);
        }

        m_outputFilecount = m_outputFiles.size();
        std::cout << "outputFilecount: " << m_outputFilecount << std::endl;

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
//            std::cout << file << std::endl;
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

        for(unsigned int itr_basis = 0; itr_basis < m_inputBasis.cols(); itr_basis++)
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
        // General/Training
        //m_fnameInputBasis = "-inputBasis_" + std::to_string(m_numberOfPrincipalModesInput) + ".bin";
        //m_fnameOutputBasis = "-outputBasis_" + std::to_string(m_numberOfPrincipalModesOutput) + ".bin";
//        m_fnameInputBasis = "-inputBasis.bin";
//        m_fnameOutputBasis = "-outputBasis.bin";

        m_pathInputMean = m_outputPrefix + "-inputMean.bin";
        m_pathInputU = m_outputPrefix + "-inputU.bin";;
        m_pathInputSigma = m_outputPrefix + "-inputSigma.bin";
        m_pathInputFeatures = m_outputPrefix + "-inputFeatures.bin";

        m_pathOutputMean = m_outputPrefix + "-outputMean.bin";
        m_pathOutputU = m_outputPrefix + "-outputU.bin";;
        m_pathOutputSigma = m_outputPrefix + "-outputSigma.bin";
        m_pathOutputFeatures = m_outputPrefix + "-outputFeatures.bin";

        // Prediction
        m_pathInputFeaturesForPrediction = m_outputPrefix + "-inputFeatures_prediction.bin";
        m_pathOutputFeaturesForPrediction = m_outputPrefix + "-outputFeatures_prediction.bin";
        m_pathGroundTruthFeatures = m_outputPrefix + "-groundtruthFeatures_prediction.bin";
    }

private:
    bool isTraining;
    bool usePrecomputed;
    bool useTestData;
    std::string m_inputPath;
    std::string m_outputPath;
    std::string m_outputPrefix;

    int m_inputFilecount;
    int m_outputFilecount;
    int m_inputVectorSize;
    int m_outputVectorSize;
    int m_nTrainingImages;
    int m_indStartTrain;
    int m_indEndTrain;

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

    VectorType m_inputSigma;
    VectorType m_outputSigma;

    MatrixType m_inputFeatures;
    MatrixType m_outputFeatures;

    TrainingPairVectorType m_trainingPairs;
    TestVectorType m_testVector;

    TestVectorType m_predictedFeatures;
    TestVectorType m_predictionVector;

    // File handling
    std::string m_fnameInputBasis;
    std::string m_fnameOutputBasis;

    std::string m_pathInputMean;
    std::string m_pathInputU;
    std::string m_pathInputSigma;
    std::string m_pathInputFeatures;
    std::string m_pathInputFeaturesForPrediction;

    std::string m_pathOutputMean;
    std::string m_pathOutputU;
    std::string m_pathOutputSigma;
    std::string m_pathOutputFeatures;
    std::string m_pathOutputFeaturesForPrediction;
    std::string m_pathGroundTruthFeatures;
};
#endif // DATAPARSER_H

