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
        m_theta = gpr::ReadMatrix<MatrixType>(filename);
    };

    void WriteModelParametersToFile(std::string filename)
    {
        gpr::WriteMatrix<MatrixType>(m_theta, filename);
    };

    MatrixType GetModelParameters()
    {
        return m_theta;
    }

    void ComputeModel(MatrixType& X, int nBatchTypes=0, int* batchSize=NULL, int* batchRepetition=NULL, bool verbose=false)
    {
        // Initialise data
        int nSamples = 0;
        int nBatches = 0;
        if(nBatchTypes == 0){
            nBatchTypes = 1;
            batchSize = new int[nBatchTypes];
            batchRepetition = new int[nBatchTypes];
            nBatches = (int)(X.rows()/m_p);
            batchSize[0] = m_p;
            batchRepetition[0] = nBatches;
        }
        else{
            for(int b=0; b<nBatchTypes; ++b) {
                nSamples += batchSize[b] * batchRepetition[b];
                nBatches += batchRepetition[b];
            }
            if(nSamples != X.rows()){
                throw std::invalid_argument("Batch parameters not correctly defined");
            }
        }

        // Compute AR model for each feature independently
        int nFeatures = X.cols();
        int K = X.rows() - nBatches;
        MatrixType theta = MatrixType::Zero(m_p, nFeatures);

        for(int f=0; f<nFeatures; ++f){
            // Declare y, D, and theta
            VectorType Y = VectorType::Zero(K);
            MatrixType D = MatrixType::Zero(K, m_p);
            if(verbose) {std::cout << "X:\n" << X << std::endl;}

            // Fill matrices
            int startInd = 0;
            int batchCount = 0;
            for(int b=0; b<nBatchTypes; ++b)
            {
                for(int rep=0; rep<batchRepetition[b]; ++rep)
                {
                    VectorType Xb = X.block(startInd,f,batchSize[b],1);
                    if(verbose) {std::cout << "Xb:\n" << Xb << std::endl;}

                    int Kb = Xb.rows() -1;
                    Y.block(startInd-batchCount,0,Kb,1) = Xb.bottomRows(Kb);
                    D.block(startInd-batchCount,0,Kb,m_p) = ComputeSubmatrix(Xb);
                    if(verbose) {std::cout << "D:\n" << D.block(startInd-batchCount,0,Kb,m_p) << std::endl;}
                    if(verbose) {std::cout << "D:\n" << D << std::endl;}

                    startInd += batchSize[b];
                    batchCount++;
                }
            }

            theta.col(f) = D.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Y);
        }

        m_theta = theta;
    }

    MatrixType Predict(MatrixType& X, int nBatchTypes=0, int* batchSize=NULL, int* batchRepetition=NULL,
                       bool onePredictionPerBatch = false, bool verbose=false)
    {
        // Initialise data
        int nSamples = 0;
        int nBatches = 0;
        if(nBatchTypes == 0){
            nBatchTypes = 1;
            batchSize = new int[nBatchTypes];
            batchRepetition = new int[nBatchTypes];
            nBatches = (int)(X.rows()/m_p);
            batchSize[0] = m_p;
            batchRepetition[0] = nBatches;
            onePredictionPerBatch = true;
        }
        else{
            for(int b=0; b<nBatchTypes; ++b) {
                nSamples += batchSize[b] * batchRepetition[b];
                nBatches += batchRepetition[b];
            }
            if(nSamples != X.rows()){
                throw std::invalid_argument("Batch parameters not correctly defined");
            }
        }

        // Predict each feature separately
        int nFeatures = X.cols();
        int K = X.rows() - nBatches;
        MatrixType YPred = MatrixType::Zero(K, nFeatures);

        for(int f=0; f<nFeatures; ++f){
            // Declare D
            MatrixType D = MatrixType::Zero(K, m_p);

            // Fill matrices (maybe as function with call by reference? --> Same logic for training and testing)
            int startInd = 0;
            int batchCount = 0;
            for(int b=0; b<nBatchTypes; ++b)
            {
                for(int rep=0; rep<batchRepetition[b]; ++rep)
                {
                    VectorType Xb = X.block(startInd,f,batchSize[b],1);
                    int Kb = Xb.rows() - 1;
                    D.block(startInd-batchCount,0,Kb,m_p) = ComputeSubmatrix(Xb);

                    startInd += batchSize[b];
                    batchCount++;
                }
            }

            // Predict n steps ahead
            MatrixType DStep = D;
            if(verbose) {std::cout << "DStep:\n" << DStep << std::endl;}
            VectorType YStep = VectorType::Zero(K);
            for(int n=0; n<m_n; ++n)
            {
                YStep = DStep*m_theta.col(f);
                MatrixType DTmp = DStep.leftCols(m_p-1);
                DStep << YStep, DTmp;
                if(verbose) {std::cout << "DStep:\n" << DStep << std::endl;}
            }
            YPred.col(f) = YStep;
        }

        if(onePredictionPerBatch){
            MatrixType YPredRed = MatrixType::Zero(nBatches, nFeatures);

            for(int b=0; b<nBatches; ++b){
                YPredRed.row(b) = YPred.row((b+1)*(batchSize[0]-1)-1);
            }
            return YPredRed;
        }

        return YPred;
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
    MatrixType m_theta;
};
#endif //AUTOREGRESSION_H
