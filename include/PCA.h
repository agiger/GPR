//
// Created by alina on 17.01.20.
//

# pragma once

#include <string>
#include <iostream>

#include "math.h"
#include "Eigen/Dense"
#include "Eigen/SVD"

#include "GaussianProcess.h"
#include "MatrixIO.h"

#ifndef PROJECT_PCA_H
#define PROJECT_PCA_H

template  <class TScalarType>
class PCA
{
public:
    typedef gpr::GaussianProcess<TScalarType>           GaussianProcessType;
    typedef std::shared_ptr<GaussianProcessType>        GaussianProcessTypePointer;
    typedef typename GaussianProcessType::VectorType    VectorType;
    typedef typename GaussianProcessType::MatrixType    MatrixType;
    typedef typename std::shared_ptr<VectorType>        VectorTypePointer;
    typedef typename std::shared_ptr<MatrixType>        MatrixTypePointer;
    typedef Eigen::JacobiSVD<MatrixType>                JacobiSVDType;
    typedef Eigen::BDCSVD<MatrixType>                   BDCSVDType;

    PCA(MatrixType& X){
        // Compute mean-free data vectors
        m_mean = X.rowwise().mean();
        MatrixType _X = X.colwise() - m_mean;

        // Compute SVD
        BDCSVDType svd(_X, Eigen::ComputeThinU);

        // Compute basis
        m_sigma = svd.singularValues()/std::sqrt((TScalarType)X.cols());
        m_U = svd.matrixU();
        m_basis = m_U*m_sigma.asDiagonal().inverse();
    }

    PCA(std::string path){
        // Read PCA matrices from files (computed based on trainings data)
        std::string fnameMean = path + "Mean.bin";
        std::string fnameSigma = path + "Sigma.bin";
        std::string fnameU = path + "U.bin";

        m_mean = gpr::ReadMatrix<MatrixType>(fnameMean);
        m_sigma = gpr::ReadMatrix<MatrixType>(fnameSigma);
        m_U = gpr::ReadMatrix<MatrixType>(fnameU);
        m_basis = m_U*m_sigma.asDiagonal().inverse();
    }

    VectorType GetMean(){
        return m_mean;
    }

    VectorType GetEigenvalues(){
        return m_sigma;
    }

    MatrixType GetMatrixU(){
        return m_U;
    }

    MatrixType GetBasis(int nFeatures=0){
        if(nFeatures > 0 && nFeatures < m_basis.cols()){
            return m_basis.leftCols(nFeatures);
        }
        else{
            return m_basis;
        }
    }

    MatrixType DimensionalityReduction(MatrixType& X, int nFeatures=0){
        MatrixType _X = X.colwise() - m_mean;
        MatrixType features = m_basis.transpose()*_X;
        if(nFeatures > 0 && nFeatures << features.rows()){
            return features.topRows(nFeatures);
        }
        else{
            return features;
        }
    }

    MatrixType GetReconstruction(MatrixType& weights){
        MatrixType basis = m_U*m_sigma.asDiagonal();
        MatrixType _X = basis.leftCols(weights.rows())*weights;
        MatrixType X = _X.colwise() + m_mean;
        return X;
    }

    VectorType GetExplainedVariance(){
        VectorType cumSum = VectorType::Zero(m_sigma.rows());
        cumSum(0) = m_sigma(0,0);
        for(int i=0; i < m_sigma.rows()-1; ++i){
            cumSum(i+1) = cumSum(i) + m_sigma(i+1, i+1);
        }
        return cumSum/cumSum.tail(1)(0);
    }

    void WriteMatricesToFile(std::string path){
        std::string fnameMean = path + "Mean.bin";
        std::string fnameSigma = path + "Sigma.bin";
        std::string fnameU = path + "U.bin";

        gpr::WriteMatrix<MatrixType>(m_mean, fnameMean);
        gpr::WriteMatrix<MatrixType>(m_sigma, fnameSigma);
        gpr::WriteMatrix<MatrixType>(m_U, fnameU);
    }

private:
    VectorType m_mean;
    VectorType m_sigma;
    MatrixType m_U;
    MatrixType m_basis;
};

#endif //PROJECT_PCA_H
