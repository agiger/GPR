/*
Output * Comment
 */

#pragma once

#include <string>
#include <iostream>
#include <algorithm>

#include <Eigen/SVD>
#include <Eigen/Dense>
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

    typedef Eigen::JacobiSVD<MatrixType>                    JacobiSVDType;
    typedef Eigen::BDCSVD<MatrixType>                       BDCSVDType;


    DataParser(std::string input_path, std::string output_path, std::string output_prefix, int input_modes, int output_modes)
    {
        isTraining = true;
        m_inputPath = input_path;
        m_outputPath = output_path;
        m_outputPrefix = output_prefix;
        m_inputFilecount = 0;
        m_outputFilecount = 0;
        m_numberOfPrincipalModesInput = input_modes;
        m_numberOfPrincipalModesOutput= output_modes;
    }

    DataParser(std::string input_path, std::string output_prefix, int input_modes, int output_modes)
    {
        isTraining = false;
        m_inputPath = input_path;
        m_outputPath = "";
        m_outputPrefix = output_prefix;
        m_inputFilecount = 0;
        m_outputFilecount = 0;
        m_numberOfPrincipalModesInput = input_modes;
        m_numberOfPrincipalModesOutput = output_modes;
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
        PcaFeatureExtractionForTraining();
        CreateTrainingVectorPair();
        return m_trainingPairs;
    }

    TestVectorType& GetTestData()
    {
        PcaFeatureExtractionForPrediction();
        CreateTestVector();
        return m_testVector;
    }

    TestVectorType GetResults(TestVectorType predicted_features)
    {
        m_testVector = predicted_features;
        inversePca();
        CreatePredictionVector();

        return m_predictionVector;
    }

protected:

    void PcaFeatureExtractionForTraining()
    {
        ParseInputFiles();
        ParseOutputFiles();
        assert(m_inputFilecount == m_outputFilecount); // use try catch instead

        ComputeInputMatrix();
        ComputeOutputMatrix();

        // Subtract Mean
        m_inputMean = m_inputMatrix.rowwise().mean();
        m_outputMean = m_outputMatrix.rowwise().mean();
        MatrixType alignedInput = m_inputMatrix.colwise() - m_inputMean;
        MatrixType alignedOutput = m_outputMatrix.colwise() - m_outputMean;

         // Computing SVD
        BDCSVDType inputSvd(alignedInput, Eigen::ComputeThinU);
        BDCSVDType outputSvd(alignedOutput, Eigen::ComputeThinU);

        // Compute Basis
        m_numberOfPrincipalModesInput = std::min(m_numberOfPrincipalModesInput, static_cast<int>(inputSvd.matrixU().cols()));
        m_numberOfPrincipalModesOutput = std::min(m_numberOfPrincipalModesOutput, static_cast<int>(outputSvd.matrixU().cols()));

        MatrixType fullInputBasis = inputSvd.matrixU() * inputSvd.singularValues().cwiseSqrt().asDiagonal();
        MatrixType fullOutputBasis = outputSvd.matrixU() * outputSvd.singularValues().cwiseSqrt().asDiagonal();

        m_inputBasis = fullInputBasis.leftCols(m_numberOfPrincipalModesInput);
        m_outputBasis = fullOutputBasis.leftCols(m_numberOfPrincipalModesOutput);

        //std::cout << std::endl;
        //std::cout << "ThinU: " << inputSvd.matrixU().rows() << "x" << inputSvd.matrixU().cols() << std::endl;
        //std::cout << "SingularValues: " << inputSvd.singularValues().rows() << "x" << inputSvd.singularValues().cols() << std::endl;
        //std::cout << "inputBasis: " << m_inputBasis.rows() << "x" << m_inputBasis.cols() << std::endl;
        //std::cout << "outputBasis: " << m_outputBasis.rows() << "x" << m_outputBasis.cols() << std::endl;
        //std::cout << "alignedInput: " << alignedInput.rows() << "x" << alignedInput.cols() << std::endl;
        //std::cout << "alignedOutput: " << alignedOutput.rows() << "x" << alignedOutput.cols() << std::endl;

        // Compute Features
        m_inputFeatures = alignedInput.transpose() * m_inputBasis;
        m_outputFeatures = alignedOutput.transpose() * m_outputBasis;

        std::cout << std::endl;
        std::cout << "m_inputFeatures: " << m_inputFeatures.rows() << "x" << m_inputFeatures.cols() << std::endl;
        std::cout << "m_outputFeatures: " << m_outputFeatures.rows() << "x" << m_outputFeatures.cols() << std::endl;

        WriteToCsvFile("inputBasis.csv", m_inputBasis);
        WriteToCsvFile("outputBasis.csv", m_outputBasis);
        WriteToCsvFile("inputMean.csv", m_inputMean);
        WriteToCsvFile("outputMean.csv", m_outputMean);

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
        ComputeInputMatrix();

        // Read input mean and basis
        std::string fname = m_outputPrefix + "-inputBasis.csv";
        m_inputBasis = ReadFromCsvFile(fname);
        std::cout << "m_inputBasis " << m_inputBasis.rows() << "x" << m_inputBasis.cols() << std::endl;

        fname = m_outputPrefix + "-inputMean.csv";
        m_inputMean = ReadFromCsvFile(fname);
        std::cout << "m_inputMean " << m_inputMean.rows() << "x" << m_inputMean.cols() << std::endl;

        // Feature extraction
        std::cout << "m_inputMatrix " << m_inputMatrix.rows() << "x" << m_inputMatrix.cols() << std::endl;
        MatrixType alignedInput = m_inputMatrix.colwise() - m_inputMean;
        m_inputFeatures = alignedInput.transpose() * m_inputBasis;

        std::cout << "m_inputFeatures " << m_inputFeatures.rows() << "x" << m_inputFeatures.cols() << std::endl;
    }

    void inversePca()
    {
        // Parse output files
        MatrixType outputFeatures(m_numberOfPrincipalModesOutput, m_testVector.size());
        uint itr = 0;
        for(const auto v : m_testVector)
        {
            std::cout << v.transpose() << std::endl;
            outputFeatures.col(itr) = v;
            itr++;
        }
        std::cout << "outputFeatures" << outputFeatures.rows() << "x" << outputFeatures.cols() << std::endl;

        // Read output mean and basis
        std::string fname = m_outputPrefix + "-outputBasis.csv";
        m_outputBasis = ReadFromCsvFile(fname);
//        std::cout << "outputBasis " << m_outputBasis.rows() << "x" << m_outputBasis.cols() << std::endl;

        fname = m_outputPrefix + "-outputMean.csv";
        m_outputMean = ReadFromCsvFile(fname);
//        std::cout << "m_outputMean " << m_outputMean.rows() << "x" << m_outputMean.cols() << std::endl;

        // inverse PCA
        MatrixType alignedOutput = m_outputBasis * outputFeatures;
//        std::cout << "alignedOutput" << alignedOutput.rows() << "x" << alignedOutput.cols() << std::endl;
        m_outputMatrix =  alignedOutput.colwise() + m_outputMean;
//        std::cout << "m_outputMatrix" << m_outputMatrix.rows() << "x" << m_outputMatrix.cols() << std::endl;
    }

    void WriteToCsvFile(std::string fname, MatrixType matrix)
    {
        std::string filename = m_outputPrefix + "-" + fname;
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
    }

    MatrixType ReadFromCsvFile(const std::string & path) {
        std::cout << "path: " << path << std::endl;
        std::ifstream indata;
        std::string line;
        std::vector<double> values;
        uint rows = 0;

        indata.open(path);
        while (std::getline(indata, line)) {
            std::stringstream lineStream(line);
            std::string cell;
            while (std::getline(lineStream, cell, ',')) {
                values.push_back(std::stod(cell));
            }
            ++rows;
        }

        uint cols = values.size()/rows;
        std::cout << "values: " << values.size() << std::endl;
        std::cout << "rows: " << rows << std::endl;
        std::cout << "cols: " << cols << std::endl;

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
        for(unsigned int itr_file; itr_file < m_outputMatrix.cols() ; ++itr_file)
        {
            VectorType v_prediction = m_outputMatrix.col(itr_file);
            m_predictionVector.push_back(v_prediction);
        }
    }

    void ComputeInputMatrix()
    {
        m_inputMatrix = MatrixType::Zero(m_inputVectorSize, m_inputFilecount);
        unsigned long counter = 0;
        for(const auto& v : m_inputFiles)
        {
            for(unsigned int itr_pix = 0; itr_pix < m_inputVectorSize; ++itr_pix)
            {
                m_inputMatrix(itr_pix, counter) = v(itr_pix);
            }
            ++counter;
        }
        return;
    }

    void ComputeOutputMatrix()
    {
        m_outputMatrix = MatrixType::Zero(m_outputVectorSize, m_outputFilecount);
        unsigned long counter = 0;
        for(const auto& v : m_outputFiles)
        {
            for(unsigned int itr_pix = 0; itr_pix < m_outputVectorSize; ++itr_pix)
            {
                m_outputMatrix(itr_pix, counter) = v(itr_pix);
            }
            ++counter;
        }
        return;
    }

    void ParseInputFiles()
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

            m_inputVectorSize = 1;
            for (unsigned itr_dim = 0; itr_dim < image_dim; ++itr_dim) {
                m_inputVectorSize *= size[itr_dim];
            }

            // Fill Eigen vector with data
            itk::ImageRegionConstIterator <TInputType> image_iterator(image, image->GetRequestedRegion());
            VectorType v = VectorType::Zero(m_inputVectorSize);

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

        return;
    }

    void ParseOutputFiles()
    {
        boost::filesystem::path file_path(m_outputPath);
        m_outputFilecount = static_cast<int>(std::distance(boost::filesystem::directory_iterator(file_path),
                                                           boost::filesystem::directory_iterator()));

        // Loop over all files
        for(unsigned int itr_file = 0; itr_file < m_outputFilecount; ++itr_file) {
            char fname[20];
            sprintf(fname, "%05d", itr_file);

            // Read data
            typename TOutputType::Pointer image = ReadImage<TOutputType>(m_outputPath + "/" + fname + ".vtk");
            typename TOutputType::SizeType size = image->GetLargestPossibleRegion().GetSize();
            static unsigned int image_dim = size.GetSizeDimension();

            m_outputVectorSize = TRANSFORM_DIMENSIONS;
            for (unsigned itr_dim = 0; itr_dim < image_dim; ++itr_dim) {
                m_outputVectorSize *= size[itr_dim];
            }

            // Fill Eigen vector with data
            itk::ImageRegionConstIterator <TOutputType> image_iterator(image, image->GetRequestedRegion());
            VectorType v = VectorType::Zero(m_outputVectorSize);

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

        return;
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

    DataVectorType m_inputFiles;
    DataVectorType m_outputFiles;

    MatrixType m_inputMatrix;
    MatrixType m_outputMatrix;

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
    TestVectorType m_predictionVector;
};

#endif // DATAPARSER_H

