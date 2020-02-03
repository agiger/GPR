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
#include "boost/filesystem.hpp"

#include <Eigen/SVD>
#include <Eigen/Dense>
#include "GaussianProcess.h"
#include "MatrixIO.h"
#include "PCA.h"
#include "AutoRegression.h"
#include "nlohmann/json.hpp"
using json = nlohmann::json;

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

    typedef itk::Image<double, IMAGE_DIMENSIONS>            ImageType;
    typedef itk::Image<itk::Vector<double, TRANSFORM_DIMENSIONS>, TRANSFORM_DIMENSIONS>     DisplacementImageType;

    typedef Eigen::JacobiSVD<MatrixType>                    JacobiSVDType;
    typedef Eigen::BDCSVD<MatrixType>                       BDCSVDType;
    typedef PCA<TScalarType>                                PcaType;
    typedef AutoRegression<TScalarType>                     AutoRegressionType;

    // Constructor for GaussianProcessLearn
    DataParser(std::string input_path, std::string output_path, std::string ar_path, std::string gpr_prefix, json config_model, json config_learn)
    {
        // Configuration parameters
        m_performAr = config_model["perform_ar"].get<bool>();
        m_usePrecomputed = config_learn["use_precomputed"].get<bool>();
        m_computeGtFeatures = false;
        m_useOnePredictionPerBatch = config_learn["ar_onePredictionPerBatch"].get<bool>();

        // Model parameters
        m_numberOfPrincipalModesInput = config_model["n_inputModes"].get<int>();
        m_numberOfPrincipalModesOutput= config_model["n_outputModes"].get<int>();
        m_n = config_model["ar_n"].get<int>();
        m_p = config_model["ar_p"].get<int>();
        std::vector<int> m_batchSizeTrain = config_learn["ar_batchSizeTrain"].get<std::vector<int>>();
        std::vector<int> m_batchRepetitionTrain = config_learn["ar_batchRepetitionTrain"].get<std::vector<int>>();
        std::vector<int> m_batchSizePredict = config_learn["ar_batchSizePredict"].get<std::vector<int>>();
        std::vector<int> m_batchRepetitionPredict = config_learn["ar_batchRepetitionPredict"].get<std::vector<int>>();
        m_nBatchTypesTrain = m_batchSizeTrain.size();
        m_nBatchTypesPredict = m_batchSizePredict.size();

        if (m_performAr){
            if (m_batchSizeTrain.size() != m_batchRepetitionTrain.size() ||
                m_batchSizePredict.size() != m_batchRepetitionPredict.size()){
                throw std::invalid_argument("AR parameters not correctly defined!");
            }
            if(m_batchSizeTrain.empty() || m_batchSizePredict.empty()){
                throw std::invalid_argument("AR parameters empty!");
            }
        }

        // File paths
        m_srcPathInput = input_path;
        m_srcPathOutput = output_path;
        m_gprPrefix = gpr_prefix;
        SetFilePaths();
        ReadFilenames(m_inputFiles, m_srcPathInput);
        ReadFilenames(m_outputFiles, m_srcPathOutput);

        // For autoregression
        if(m_performAr){
            boost::filesystem::path root(ar_path);
            boost::filesystem::path train("train");
            boost::filesystem::path test("test");
            boost::filesystem::path path_train = root / train;
            boost::filesystem::path path_test = root / test;

            ReadFilenames(m_arFilesTrain, path_train.string());
            ReadFilenames(m_arFilesTest, path_test.string());
        }

        // Use subset of training data only (e.g. for drift analysis)
        m_nTrainingImages = config_learn["n_trainImgs"].get<int>();
        m_indStartTrain = config_learn["start_trainInd"].get<int>();;
        m_indEndTrain = m_indStartTrain + m_nTrainingImages -1;
        if(m_nTrainingImages != 0)
        {
            std::cout << "Only a subset of the training data is considered:" << std::endl;
            std::cout << "indStart: " << m_indStartTrain << std::endl;
            std::cout << "indEnd: " << m_indEndTrain << std::endl;
            std::cout << "nImgs: " << m_nTrainingImages << std::endl;

            if(m_performAr){
                m_inputFiles.erase(m_inputFiles.begin()+(m_indEndTrain+1)*m_p, m_inputFiles.end());
                m_inputFiles.erase(m_inputFiles.begin(), m_inputFiles.begin()+m_indStartTrain*m_p);
            }
            else{
                m_inputFiles.erase(m_inputFiles.begin()+m_indEndTrain+1, m_inputFiles.end());
                m_inputFiles.erase(m_inputFiles.begin(), m_inputFiles.begin()+m_indStartTrain);
            }
            m_outputFiles.erase(m_outputFiles.begin()+m_indEndTrain+1, m_outputFiles.end());
            m_outputFiles.erase(m_outputFiles.begin(), m_outputFiles.begin()+m_indStartTrain);
        }
    }

    // Constructor for GaussianProcessPredict
    DataParser(std::string input_path, std::string groundtruth_path, std::string gpr_prefix, json config_model, json config_predict)
    {
        // Configuration parameters
        m_performAr = config_model["perform_ar"].get<bool>();
        m_usePrecomputed = config_predict["use_precomputed"].get<bool>();
        m_computeGtFeatures = config_predict["compute_groundtruth_features"].get<bool>();
        m_useOnePredictionPerBatch = config_predict["ar_onePredictionPerBatch"].get<bool>();

        // Model parameters
        m_numberOfPrincipalModesInput = config_model["n_inputModes"].get<int>();
        m_numberOfPrincipalModesOutput= config_model["n_outputModes"].get<int>();
        m_n = config_model["ar_n"].get<int>();
        m_p = config_model["ar_p"].get<int>();
        m_batchSizePredict = config_predict["ar_batchSizePredict"].get<std::vector<int>>();
        m_batchRepetitionPredict = config_predict["ar_batchRepetitionPredict"].get<std::vector<int>>();
        m_nBatchTypesPredict = m_batchSizePredict.size();

        if (m_performAr){
            if (m_batchSizePredict.size() != m_batchRepetitionPredict.size()){
                throw std::invalid_argument("AR parameters not correctly defined!");
            }
            if(m_batchSizePredict.empty()){
                throw std::invalid_argument("AR parameters empty!");
            }
        }

        // File paths
        m_srcPathInput = input_path;
        m_srcPathOutput = groundtruth_path;
        m_gprPrefix = gpr_prefix;
        SetFilePaths();
        ReadFilenames(m_inputFiles, m_srcPathInput);
        ReadFilenames(m_outputFiles, m_srcPathOutput);
    }

    ~DataParser(){}

    TrainingPairVectorType GetTrainingData()
    {
        PcaFeatureExtractionForTraining();
        std::cout << "inputFeatures: " << m_inputFeatures.rows() << "x" << m_inputFeatures.cols() << std::endl;
        std::cout << "outputFeatures: " << m_outputFeatures.rows() << "x" << m_outputFeatures.cols() << std::endl;
        CreateTrainingVectorPair();
        return m_trainingPairs;
    }

    TestVectorType GetTestData()
    {
        PcaFeatureExtractionForPrediction();
        std::cout << "inputFeatures: " << m_inputFeatures.rows() << "x" << m_inputFeatures.cols() << std::endl;
        std::cout << "outputFeatures: " << m_outputFeatures.rows() << "x" << m_outputFeatures.cols() << std::endl;
        CreateTestVector();
        return m_testVector;
    }

    TestVectorType GetResults(TestVectorType predicted_features)
    {
        m_predictedFeatures = predicted_features;
        inversePca();
        CreatePredictionVector();
        return m_predictionVector;
    }

protected:
    void ComputeFeaturesForTraining(MatrixType& features, MatrixType& matrix, int nModes, std::string prefix, std::string reference)
    {
        // Compute PCA
        auto t0 = std::chrono::system_clock::now();
        PcaType pca(matrix);
        std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-t0;
        std::cout << "Pca done in " << elapsed_seconds.count() << "s" << std::endl;

        // Features
        features = pca.DimensionalityReduction(matrix, nModes);

        // Save mean and basis as images
        VectorType mean = pca.GetMean();
        MatrixType basis = pca.GetBasis(nModes);
        WriteVectorToImage(mean, reference, prefix + "Mean.vtk");
        WriteMatrixToImageSeries(basis, reference, prefix);

        // Explained Variance
        VectorType explainedVar= pca.GetExplainedVariance();
        gpr::WriteMatrix<MatrixType>(explainedVar, prefix + "Compactness.bin");

        // Write files
        pca.WriteMatricesToFile(prefix);

        MatrixType fullFeatures = pca.DimensionalityReduction(matrix);
        gpr::WriteMatrix<MatrixType>(fullFeatures, prefix + "Features.bin");

        std::cout << "Basis: " << basis.rows() << "x" << basis.cols() << std::endl;
        std::cout << "Features: " << features.rows() << "x" << features.cols() << std::endl;

        return;
    }

    void PcaFeatureExtractionForTraining()
    {
        if(!m_usePrecomputed)
        {
            ParseImageFiles(m_inputMatrix, m_inputFiles);
            ParseDisplacementFiles(m_outputMatrix, m_outputFiles);
            if(m_inputMatrix.cols() % m_outputMatrix.cols() != 0){
                throw std::invalid_argument("Wrong number of input or output files");
            }

            ComputeFeaturesForTraining(m_outputFeatures, m_outputMatrix, m_numberOfPrincipalModesOutput, m_gprPrefixOutput, m_outputFiles.front());
            if(m_performAr){
                ComputeFeaturesForTraining(m_inputFeatures, m_inputMatrix, m_numberOfPrincipalModesInput, m_gprPrefixInput, m_inputFiles.front());
            }
            else{
                // AR: Autoregression
                ParseImageFiles(m_arMatrixTrain, m_arFilesTrain);
                ParseImageFiles(m_arMatrixTest, m_arFilesTest);

                // Concatenate Matrices & Features
                MatrixType concatArMatrix(m_arMatrixTrain.rows(), m_arMatrixTrain.cols() + m_arMatrixTest.cols());
                concatArMatrix.leftCols(m_arMatrixTrain.cols()) = m_arMatrixTrain;
                concatArMatrix.rightCols(m_arMatrixTest.cols()) = m_arMatrixTest;

                MatrixType concatInputMatrix(m_inputMatrix.rows(), m_inputMatrix.cols() +  concatArMatrix.cols());
                concatInputMatrix.leftCols(m_inputMatrix.cols()) = m_inputMatrix;
                concatInputMatrix.rightCols(concatArMatrix.cols()) = concatArMatrix;

                // Compute Features for concatenated input matrix
                MatrixType concatInputFeatures;
                ComputeFeaturesForTraining(concatInputFeatures, concatInputMatrix, m_numberOfPrincipalModesInput, m_gprPrefixInput, m_inputFiles.front());

                // Split Features
                MatrixType inputFeaturesTranspose = concatInputFeatures.leftCols(m_inputMatrix.cols()).transpose();
                MatrixType concatArFeatures = concatInputFeatures.rightCols(concatArMatrix.cols());
                MatrixType arFeaturesTrainTranspose = concatArFeatures.leftCols(m_arMatrixTrain.cols()).transpose();
                MatrixType arFeaturesTestTranspose = concatArFeatures.rightCols(m_arMatrixTest.cols()).transpose();

                // Compute AR model
                AutoRegressionType ar(m_n, m_p);
                ar.ComputeModel(arFeaturesTrainTranspose, m_nBatchTypesTrain, &m_batchSizeTrain[0], &m_batchRepetitionTrain[0]);
                ar.WriteModelParametersToFile(m_gprPrefix + "-arModel.bin");
                m_inputFeatures = ar.Predict(inputFeaturesTranspose).transpose();
                MatrixType arFeaturesTestPredictTranspose = ar.Predict(arFeaturesTestTranspose, m_nBatchTypesPredict, &m_batchSizePredict[0], &m_batchRepetitionPredict[0], m_useOnePredictionPerBatch);

                gpr::WriteMatrix<MatrixType>(arFeaturesTestTranspose, m_gprPrefix + "-arFeaturesTest.bin");
                gpr::WriteMatrix<MatrixType>(arFeaturesTestPredictTranspose, m_gprPrefix + "-arFeaturesTestPredict.bin");
            }
        }
        else
        {
            // Read output features
            MatrixType fullOutputFeatures = gpr::ReadMatrix<MatrixType>(m_gprPrefixOutput + "Features.bin");
            m_outputFeatures = fullOutputFeatures.topRows(m_numberOfPrincipalModesOutput);

            if(m_performAr){
                // Read input features
                MatrixType fullInputFeatures = gpr::ReadMatrix<MatrixType>(m_gprPrefixInput + "Features.bin");
                m_inputFeatures = fullInputFeatures.topRows(m_numberOfPrincipalModesInput);
            }
            else{
                MatrixType fullInputFeatures = gpr::ReadMatrix<MatrixType>(m_gprPrefixInput + "Features.bin");
                MatrixType concatInputFeatures = fullInputFeatures.topRows(m_numberOfPrincipalModesInput);
                MatrixType inputFeaturesTranspose = concatInputFeatures.leftCols(m_inputFiles.size()).transpose();

                // Autoregression: predict input features
                AutoRegressionType ar(m_n, m_p);
                ar.ReadModelParametersFromFile(m_gprPrefix + "-arModel.bin");
                m_inputFeatures = ar.Predict(inputFeaturesTranspose, m_nBatchTypesPredict, &m_batchSizePredict[0], &m_batchRepetitionPredict[0], m_useOnePredictionPerBatch).transpose();
            }
        }

    }

    void PcaFeatureExtractionForPrediction()
    {
        // Compute/Read input features
        if(!m_usePrecomputed)
        {
            // Parse input files
            ParseImageFiles(m_inputMatrix, m_inputFiles);
            PcaType inputPca(m_gprPrefixInput);

            MatrixType fullInputFeatures = inputPca.DimensionalityReduction(m_inputMatrix);
            gpr::WriteMatrix<MatrixType>(fullInputFeatures, m_pathInputFeaturesForPrediction);

            if(m_performAr){
                m_inputFeatures = inputPca.DimensionalityReduction(m_inputMatrix, m_numberOfPrincipalModesInput);
            }
            else{
                MatrixType inputFeaturesTranspose = inputPca.DimensionalityReduction(m_inputMatrix, m_numberOfPrincipalModesInput).transpose();

                // Autoregression: predict input features
                AutoRegressionType ar(m_n, m_p);
                ar.ReadModelParametersFromFile(m_gprPrefix + "-arModel.bin");
                m_inputFeatures = ar.Predict(inputFeaturesTranspose, m_nBatchTypesPredict, &m_batchSizePredict[0], &m_batchRepetitionPredict[0], m_useOnePredictionPerBatch).transpose();
            }
        }
        else
        {
            MatrixType fullInputFeatures = gpr::ReadMatrix<MatrixType>(m_pathInputFeaturesForPrediction);
            if(m_performAr){
                m_inputFeatures = fullInputFeatures.topRows(m_numberOfPrincipalModesInput);
            }
            else{
                MatrixType inputFeaturesTranspose = fullInputFeatures.topRows(m_numberOfPrincipalModesInput).transpose();

                // Autoregression: predict input features
                AutoRegressionType ar(m_n, m_p);
                ar.ReadModelParametersFromFile(m_gprPrefix + "-arModel.bin");
                m_inputFeatures = ar.Predict(inputFeaturesTranspose, m_nBatchTypesPredict, &m_batchSizePredict[0], &m_batchRepetitionPredict[0], m_useOnePredictionPerBatch).transpose();
            }
        }

        // Compute + write ground truth features, if required
        if(!m_computeGtFeatures)
        {
            // Parse ground truth files
            ParseDisplacementFiles(m_outputMatrix, m_outputFiles);
            PcaType outputPca(m_gprPrefixOutput);

            m_outputFeatures = outputPca.DimensionalityReduction(m_outputMatrix, m_numberOfPrincipalModesOutput);
            MatrixType m_fullOutputFeatures = outputPca.DimensionalityReduction(m_outputMatrix);
            gpr::WriteMatrix<MatrixType>(m_fullOutputFeatures, m_pathGroundTruthFeatures);
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
        PcaType outputPca(m_gprPrefixOutput);
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

    void ReadFilenames(std::vector<std::string>& files, std::string path)
    {
        for(const auto & p : std::experimental::filesystem::directory_iterator(path))
        {
            files.push_back(p.path().string());
        }
        sort(files.begin(), files.end());
        std::cout << "Filecount: " << files.size() << std::endl;
    }

    void ParseImageFiles(MatrixType& matrix, std::vector<std::string>& filenames)
    {
        // Identify vector size
        typename TInputType::Pointer image = ReadImage<TInputType>(filenames.front());
        typename TInputType::SizeType size = image->GetLargestPossibleRegion().GetSize();
        static unsigned int image_dim = size.GetSizeDimension();

        int imageVectorSize = 1;
        for (unsigned itr_dim = 0; itr_dim < image_dim; ++itr_dim) {
            imageVectorSize *= size[itr_dim];
        }

        // Loop over all files
        unsigned int counter_file = 0;
        matrix = MatrixType::Zero(imageVectorSize, filenames.size());
        for(std::string & file : filenames)
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
                matrix(counter_pix, counter_file) = (TScalarType)pixel/(TScalarType)255;
                ++counter_pix;
                ++image_iterator;
            }
            ++counter_file;
        } // end for all files

        return;
    }

    void ParseDisplacementFiles(MatrixType& matrix, std::vector<std::string>& filenames)
    {
        // Identify vector size
        typename TOutputType::Pointer image = ReadImage<TOutputType>(filenames.front());
        typename TOutputType::SizeType size = image->GetLargestPossibleRegion().GetSize();
        static unsigned int image_dim = size.GetSizeDimension();

        int displacementVectorSize = TRANSFORM_DIMENSIONS;
        for (unsigned itr_dim = 0; itr_dim < image_dim; ++itr_dim) {
            displacementVectorSize *= size[itr_dim];
        }

        // Loop over all files
        unsigned int counter_file = 0;
        matrix = MatrixType::Zero(displacementVectorSize, filenames.size());
        for(std::string & file : filenames)
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
                    matrix(counter_pix, counter_file) = (TScalarType)pixel[itr_df];
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
        m_gprPrefixInput = m_gprPrefix + "-input";
        m_gprPrefixOutput = m_gprPrefix + "-output";

        // Training
        m_pathInputFeatures = m_gprPrefixInput + "Features.bin";
        m_pathOutputFeatures = m_gprPrefixOutput + "Features.bin";

        // Prediction
        m_pathInputFeaturesForPrediction = m_gprPrefixInput + "Features_prediction.bin";
        m_pathOutputFeaturesForPrediction = m_gprPrefixOutput + "Features_prediction.bin";
        m_pathGroundTruthFeatures = m_gprPrefix + "-groundtruthFeatures_prediction.bin";
    }

private:
    // Configuration parameters
    bool m_performAr;
    bool m_usePrecomputed;
    bool m_computeGtFeatures;
    bool m_useOnePredictionPerBatch;

    // Model parameters
    int m_numberOfPrincipalModesInput;
    int m_numberOfPrincipalModesOutput;
    int m_n, m_p;
    int m_nBatchTypesTrain, m_nBatchTypesPredict;
    std::vector<int> m_batchSizeTrain, m_batchRepetitionTrain;
    std::vector<int> m_batchSizePredict, m_batchRepetitionPredict;


    // Use subset of training data only (e.g. for drift analysis)
    int m_nTrainingImages;
    int m_indStartTrain;
    int m_indEndTrain;

    // File paths
    std::string m_srcPathInput;
    std::string m_srcPathOutput;
    std::string m_gprPrefix;
    std::string m_gprPrefixInput;
    std::string m_gprPrefixOutput;

    std::string m_pathInputFeatures;
    std::string m_pathOutputFeatures;
    std::string m_pathInputFeaturesForPrediction;
    std::string m_pathOutputFeaturesForPrediction;
    std::string m_pathGroundTruthFeatures;

    std::vector<std::string> m_inputFiles;
    std::vector<std::string> m_outputFiles;
    std::vector<std::string> m_arFilesTrain;
    std::vector<std::string> m_arFilesTest;

    // Data matrices
    MatrixType m_inputMatrix;
    MatrixType m_outputMatrix;
    MatrixType m_arMatrixTrain;
    MatrixType m_arMatrixTest;
    MatrixType m_predictedOutputMatrix;
    MatrixType m_inputFeatures;
    MatrixType m_outputFeatures;

    // Output structures
    TrainingPairVectorType m_trainingPairs;
    TestVectorType m_testVector;
    TestVectorType m_predictedFeatures;
    TestVectorType m_predictionVector;
};
#endif // DATAPARSER_H

