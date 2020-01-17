//
// Created by alina on 15.01.20.
//

# pragma once

#include <string>
#include <iostream>
#include <numeric>

#include "Eigen/Dense"
#include "GaussianProcess.h"
#include "MatrixIO.h"

#ifndef AUTOREGRESSION_H
#define AUTOREGRESSION_H


template <class TScalarType>
class AutoRegression
{
public:
    typedef gpr::GaussianProcess<TScalarType>           GaussianProcessType;
    typedef std::shared_ptr<GaussianProcessType>        GaussianProcessTypePointer;
    typedef typename GaussianProcessType::VectorType    VectorType;
    typedef typename GaussianProcessType::MatrixType    MatrixType;
    typedef typename std::shared_ptr<VectorType>        VectorTypePointer;
    typedef typename std::shared_ptr<MatrixType>        MatrixTypePointer;

    AutoRegression(int n, int p)
    {
        m_n = n;
        m_p = p;
    }

    void ReadModelParametersFromFile(std::string filename)
    {
       m_theta = gpr::ReadMatrix<VectorType>(filename);
    };

    void WriteModelParametersToFile(std::string filename)
    {
//        MatrixType tmp(m_theta.rows(), m_theta.cols()*2);
//        tmp << m_theta, m_theta;

        // Writing does not work properly
//        gpr::WriteMatrix<VectorType>(m_theta, filename);
    };



    void ComputeModel(VectorType& X, int nBatchTypes=1, int* batchSize=NULL, int* batchRepetition=NULL)
    {
//        if(verbose) {std::cout << "X:\n" << X << std::endl;}
//        if(verbose) {std::cout << "Y:\n" << Y << std::endl;}
//        if(verbose) {std::cout << "D:\n" << D << std::endl;}
        // Initialise data
        int nSamples = 0;
        int nBatches = 0;
        if(nBatchTypes == 1){
            batchSize = new int[nBatchTypes];
            batchRepetition = new int[nBatchTypes];
            batchSize[0] = X.rows();
            batchRepetition[0] = 1;
            nBatches = 1;
        }
        else{
            for(int b=0; b<nBatchTypes; ++b) {
                nSamples += batchSize[b] * batchRepetition[b];
                nBatches += batchRepetition[b];
            }
            if(nSamples != X.rows()){
                throw std::string("Batch parameters not correctly defined");
            }
        }

        // Declare y, D, and theta
        int K = X.rows() - nBatches;
        VectorType Y = VectorType::Zero(K);
        MatrixType D = MatrixType::Zero(K, m_p);
        std::cout << "X:\n" << X << std::endl;

        // Fill matrices
        int startInd = 0;
        int batchCount = 0;
        for(int b=0; b<nBatchTypes; ++b)
        {
            for(int rep=0; rep<batchRepetition[b]; ++rep)
            {
                VectorType Xb = X.block(startInd,0,batchSize[b],1);
                std::cout << "Xb:\n" << Xb << std::endl;

                int Kb = Xb.rows() -1;
                Y.block(startInd-batchCount,0,Kb,1) = Xb.bottomRows(Kb);
                D.block(startInd-batchCount,0,Kb,m_p) = ComputeSubmatrix(Xb);
                std::cout << "D:\n" << D.block(startInd-batchCount,0,Kb,m_p) << std::endl;
                std::cout << "D:\n" << D << std::endl;

                startInd += batchSize[b];
                batchCount++;
            }
        }

        m_theta = D.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Y);

//        int K = X.rows() - 1;
//        m_theta = VectorType::Zero(m_p);

//        MatrixType D = MatrixType::Zero(K, m_p);
//        VectorType Y = X.bottomRows(K);
//        for(int k=0; k<m_p; ++k)
//        {
//            D.bottomRows(K-k).col(k) = X.topRows(K-k);
//            m_theta = D.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Y);
//        }
    }

    VectorType Predict(VectorType& X, int nBatchTypes=1, int* batchSize=NULL, int* batchRepetition=NULL)
    {
        // Initialise data
        int nSamples = 0;
        int nBatches = 0;
        if(nBatchTypes == 1){
            batchSize = new int[nBatchTypes];
            batchRepetition = new int[nBatchTypes];
            batchSize[0] = X.rows();
            batchRepetition[0] = 1;
            nBatches = 1;
        }
        else{
            for(int b=0; b<nBatchTypes; ++b) {
                nSamples += batchSize[b] * batchRepetition[b];
                nBatches += batchRepetition[b];
            }
            if(nSamples != X.rows()){
                throw std::string("Batch parameters not correctly defined");
            }
        }

        // Declare D
        int K = X.rows() - nBatches;
        MatrixType D = MatrixType::Zero(K, m_p);

        // Fill matrices (maybe as function with call by reference? --> Same logic for training and testing)
        int startInd = 0;
        int batchCount = 0;
        for(int b=0; b<nBatchTypes; ++b)
        {
            for(int rep=0; rep<batchRepetition[b]; ++rep)
            {
                VectorType Xb = X.block(startInd,0,batchSize[b],1);
                int Kb = Xb.rows() - 1;
                D.block(startInd-batchCount,0,Kb,m_p) = ComputeSubmatrix(Xb);

                startInd += batchSize[b];
                batchCount++;
            }
        }

        // Predict n steps ahead
        MatrixType DStep = D;
        std::cout << "DStep:\n" << DStep << std::endl;
        VectorType YStep = VectorType::Zero(K);
        for(int n=0; n<m_n; ++n)
        {
            YStep = DStep*m_theta;
            MatrixType DTmp = DStep.leftCols(m_p-1);
            DStep << YStep, DTmp;
            std::cout << "DStep:\n" << DStep << std::endl;
        }


//        int K = X.rows() - 1;

//        MatrixType D = MatrixType::Zero(K, m_p);
//        for(int k=0; k<m_p; ++k) {
//            D.bottomRows(K - k).col(k) = X.topRows(K - k);
//        }

//        // Predict n steps ahead:
//        MatrixType DStep = D;
//        VectorType YStep = VectorType::Zero(K);
//        for(int n=0; n<m_n; ++n)
//        {
//            YStep = DStep*m_theta;
//            MatrixType DTmp = DStep.leftCols(m_p-1);
//            DStep << YStep, DTmp;
//        }

        return YStep;
    }

    bool AutoRegressionTest()
    {
        int nBatchTypes = 2;
        int batchSize[nBatchTypes] = {2, 4};
        int batchRepetition[nBatchTypes] = {1, 1};
        VectorType X(6);
        X << 1, 2, 3, 4, 5, 6;
        VectorType X_test(6);
        X_test << 7, 8, 9, 10, 11, 12;

        std::cout << "Compute model 1" << std::endl;
        ComputeModel(X, nBatchTypes, batchSize, batchRepetition);
        std::cout << "Predict model 1" << std::endl;
        VectorType YPred1 = Predict(X_test, true);
        std::cout << "theta:\n" << m_theta << std::endl;
        std::cout << "YPred:\n" << YPred1 << std::endl;

        std::cout << "Compute model 2" << std::endl;
        ComputeModel(X);
        std::cout << "Predict model 2" << std::endl;
        VectorType YPred2 = Predict(X_test, true);
        std::cout << "theta:\n" << m_theta << std::endl;
        std::cout << "YPred:\n" << YPred2 << std::endl;

//        WriteModelParametersToFile("/tmp/test_theta.txt");

        return true;
    }

protected:
    MatrixType ComputeSubmatrix(VectorType& X)
    {
        int K = X.rows() - 1;
        MatrixType D = MatrixType::Zero(K, m_p);

        for(int k=0; k<m_p; ++k){
            D.bottomRows(K-k).col(k) = X.topRows(K-k);
        }

        return D;
    }

private:
    int m_n;          // n-step ahead prediction
    int m_p;          // order of AR model
    VectorType m_theta;
};
#endif //AUTOREGRESSION_H
