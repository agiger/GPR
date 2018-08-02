/*
 * Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation (CIAN)
 *
 * Author: Robin Sandkuehler (robin.sandkuehler@unibas.ch)
 * Author: Christoph Jud (christoph.jud@unibas.ch)
 *
 */
#pragma once

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkTransformToDisplacementFieldFilter.h>
#include <itkThresholdImageFilter.h>
#include <itkStatisticsImageFilter.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>
#include <itkMedianImageFilter.h>
#include <itkHistogramMatchingImageFilter.h>
#include <itkWarpImageFilter.h>
#include <itkBSplineInterpolateImageFunction.h>
#include <itkGradientAnisotropicDiffusionImageFilter.h>
#include <itkCurvatureAnisotropicDiffusionImageFilter.h>
#include <itkDisplacementFieldTransform.h>
#include "itkDiscreteGaussianImageFilter.h"
#include <itkAbsoluteValueDifferenceImageFilter.h>
#include <itkMeanImageFilter.h>
#include <itkVectorMagnitudeImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkCurvatureFlowImageFilter.h>
#include <itkNormalizeImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkImageSeriesWriter.h>
#include <itkImageSeriesReader.h>
#include <itkPNGImageIO.h>
#include <itkVTKImageIO.h>
#include <itkNumericSeriesFileNames.h>
#include <itkMultiplyImageFilter.h>
#include "itkRegionOfInterestImageFilter.h"
#include "itkJoinSeriesImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkVectorResampleImageFilter.h"
#include "itkShrinkImageFilter.h"
#include "itkInvertIntensityImageFilter.h"
#include "itkStatisticsImageFilter.h"
#include "itkImage.h"
#include <itkRecursiveMultiResolutionPyramidImageFilter.h>
#include "itkImageDuplicator.h"
#include "itkRoundImageFilter.h"
#include "itkSquareImageFilter.h"
#include <itkSubtractImageFilter.h>
#include "itkThresholdImageFilter.h"

#include <omp.h>

//#include "gprDataTypes.h"
typedef double ScalarType;

#include <utility>

template<typename TImageType>
typename TImageType::Pointer CopyImage(typename TImageType::Pointer image)
{
    typedef itk::ImageDuplicator<TImageType> DuplicatorType;
    auto duplicator = DuplicatorType::New();
    duplicator->SetInputImage(image);
    duplicator->Update();
    return duplicator->GetOutput();
}

template<typename TImageType>
typename TImageType::Pointer RoundImage(typename TImageType::Pointer image)
{
    typedef itk::RoundImageFilter<TImageType, TImageType> FilterType;
    auto filter = FilterType::New();
    filter->SetInput(image);
    filter->Update();
    return filter->GetOutput();
}

template<typename TImageType>
typename TImageType::Pointer ThresholdImage(typename TImageType::Pointer image, ScalarType threshold)
{
    typedef itk::ThresholdImageFilter<TImageType> FilterType;
    auto filter = FilterType::New();
    filter->SetInput(image);
    filter->SetOutsideValue(0);
    filter->ThresholdBelow(threshold);
    filter->Update();
    return filter->GetOutput();
}

template<typename TImageType>
typename TImageType::Pointer NewImage2dImage(int width, int height);

template<typename TImageType>
typename TImageType::Pointer NewImage3dImage(typename TImageType::SizeType size)
{
    typename TImageType::RegionType region;
    typename TImageType::IndexType start;
    start[0] = 0;
    start[1] = 0;
    start[2] = 0;

    region.SetSize(size);
    region.SetIndex(start);

    typename TImageType::Pointer image = TImageType::New();
    image->SetRegions(region);
    image->Allocate();
    image->FillBuffer(0);

    return image;
}

template<typename TImageType>
typename TImageType::Pointer GetTargetImageFromImageSeries(typename TImageType::Pointer imageSeries)
{
    typename TImageType::SizeType imageSize;

    imageSize = imageSeries->GetLargestPossibleRegion().GetSize();

    int numberOfPixelSingleImage = imageSize[0]*imageSize[1];
    int numberOfimages = imageSize[2];
    std::vector<ScalarType> imageMeans;

    ScalarType *pixelBuffer = imageSeries->GetBufferPointer();

    for(int i = 0; i < numberOfimages; ++i)
    {
        ScalarType mean(0);
        for(int j = 0; j < numberOfPixelSingleImage; ++j)
        {
            mean += pixelBuffer[i*numberOfPixelSingleImage + j];
        }

        imageMeans.push_back(mean/numberOfPixelSingleImage);
    }

    //compute over all mean
    ScalarType overAllMean(0);
    for(int i = 0; i < imageMeans.size(); ++i)
    {
        overAllMean += imageMeans.at(i);
    }

    overAllMean /= imageMeans.size();

    //find best image close to the over all mean

    int targetID = 0;
    ScalarType meanDiff(1e10);
    for(int i = 0; i < numberOfimages; ++i)
    {
        if(abs(imageMeans.at(i) - overAllMean) < meanDiff)
        {
            targetID = i;
            meanDiff = abs(imageMeans.at(i) - overAllMean) ;
        }
    }

    auto targetImageSeries = NewImage3dImage<TImageType>(imageSize);

    ScalarType *targetImageBuffer = targetImageSeries->GetBufferPointer();
    for(int i = 0; i < numberOfimages; ++i)
        std::memcpy(&targetImageBuffer[i*numberOfPixelSingleImage],
                &pixelBuffer[targetID*numberOfPixelSingleImage],
                numberOfPixelSingleImage*sizeof(ScalarType));

    targetImageSeries->SetSpacing(imageSeries->GetSpacing());
    targetImageSeries->SetOrigin(imageSeries->GetOrigin());

    return targetImageSeries;
}

template<typename TImageType>
std::vector<typename TImageType::Pointer> GetImagePyramideImage(typename TImageType::Pointer image,
                                                                int numberOfScales)
{
    typedef itk::RecursiveMultiResolutionPyramidImageFilter<TImageType, TImageType> PyramidImageFilterType;

    auto imagePyramide = PyramidImageFilterType::New();
    imagePyramide->SetInput(image);
    imagePyramide->SetNumberOfLevels(numberOfScales);
    imagePyramide->Update();

    std::vector<typename TImageType::Pointer> images;
    for(int i = 0; i < numberOfScales; ++i)
    {
        images.push_back(imagePyramide->GetOutput(i));
    }

    // set the last pyramide level to the original data
    images[numberOfScales - 1] = image;

    return images;
}

template<typename TImageType, typename TImageSliceType>
std::vector<typename TImageType::Pointer> GetImagePyramideImageSeriesTarget(typename TImageType::Pointer imageSeries,
                                                                            int numberOfScales)
{
    typename TImageType::SizeType imageSeriesSize = imageSeries->GetLargestPossibleRegion().GetSize();
    typename TImageType::SpacingType imageSeriesSpacing = imageSeries->GetSpacing();
    typename TImageType::PointType imageSeriesOrigin = imageSeries->GetOrigin();

    typename TImageSliceType::SizeType imageBufferSize;
    typename TImageSliceType::SpacingType imageBufferSpacing;
    typename TImageSliceType::PointType imageBufferOrigin;

    for(int d = 0; d < IMAGE_DIMENSIONS - 1; ++d)
    {
        imageBufferSize[d] = imageSeriesSize[d];
        imageBufferSpacing[d] = imageSeriesSpacing[d];
        imageBufferOrigin[d] = imageSeriesOrigin[d];
    }

    auto sliceImageBuffer = NewImage2dImage<TImageSliceType>(imageBufferSize[0], imageBufferSize[1]);
    sliceImageBuffer->SetSpacing(imageBufferSpacing);
    sliceImageBuffer->SetOrigin(imageBufferOrigin);

    std::memcpy(sliceImageBuffer->GetBufferPointer(), imageSeries->GetBufferPointer(), imageBufferSize[0]*imageBufferSize[1]*sizeof(ScalarType));

    typedef itk::RecursiveMultiResolutionPyramidImageFilter<TImageSliceType, TImageSliceType> PyramidImageFilterType;

    auto imagePyramide = PyramidImageFilterType::New();
    imagePyramide->SetInput(sliceImageBuffer);
    imagePyramide->SetNumberOfLevels(numberOfScales);
    imagePyramide->Update();

    // create image pyramides
    std::vector<typename TImageType::SizeType> imageSizes;
    std::vector<typename TImageType::Pointer> images;
    for(int i = 0; i < numberOfScales; ++i)
    {
        auto image = imagePyramide->GetOutput(i);
        auto imageSizeTMP = image->GetLargestPossibleRegion().GetSize();
        typename TImageType::SizeType sizeTMP;

        for(int d = 0; d < IMAGE_DIMENSIONS - 1; ++d)
            sizeTMP[d] = imageSizeTMP[d];

        sizeTMP[IMAGE_DIMENSIONS - 1] = imageSeriesSize[IMAGE_DIMENSIONS - 1];

        imageSizes.push_back(sizeTMP);

        auto imagePyramideTMP = NewImage3dImage<TImageType>(imageSizes[i]);
        typename TImageType::SpacingType spacingTMP;
        typename TImageType::PointType originTMP;

        for(int d = 0; d < IMAGE_DIMENSIONS - 1; ++d)
        {
            spacingTMP[d] = image->GetSpacing()[d];
            originTMP[d] = image->GetOrigin()[d];
        }

        spacingTMP[IMAGE_DIMENSIONS - 1] = imageSeriesSpacing[IMAGE_DIMENSIONS - 1];
        originTMP[IMAGE_DIMENSIONS - 1] = imageSeriesOrigin[IMAGE_DIMENSIONS - 1];

        imagePyramideTMP->SetSpacing(spacingTMP);
        imagePyramideTMP->SetOrigin(originTMP);
        images.push_back(imagePyramideTMP);
    }

    // create image pyramide for all images
    for(int i = 0; i < imageSeriesSize[2]; ++i)
    {
        //atach image to pyrmide syries
        for(int scaleIndex = 0; scaleIndex < numberOfScales; ++scaleIndex)
        {
            auto imageSize = imageSizes[scaleIndex];
            int numberOfPixels = imageSize[0]*imageSize[1];
            auto imageTMP = imagePyramide->GetOutput(scaleIndex);
            std::memcpy(&images[scaleIndex]->GetBufferPointer()[i*numberOfPixels], imageTMP->GetBufferPointer(),
                    numberOfPixels*sizeof(ScalarType));
        }
    }
    // set the last pyramide level to the original data
    images[numberOfScales - 1] = imageSeries;

    return images;
}


template<typename TImageType, typename TImageSliceType>
std::vector<typename TImageType::Pointer> GetImagePyramideImageSeries(typename TImageType::Pointer imageSeries,
                                                                      int numberOfScales)
{
    typename TImageType::SizeType imageSeriesSize = imageSeries->GetLargestPossibleRegion().GetSize();
    typename TImageType::SpacingType imageSeriesSpacing = imageSeries->GetSpacing();
    typename TImageType::PointType imageSeriesOrigin = imageSeries->GetOrigin();

    typename TImageSliceType::SizeType imageBufferSize;
    typename TImageSliceType::SpacingType imageBufferSpacing;
    typename TImageSliceType::PointType imageBufferOrigin;

    for(int d = 0; d < IMAGE_DIMENSIONS - 1; ++d)
    {
        imageBufferSize[d] = imageSeriesSize[d];
        imageBufferSpacing[d] = imageSeriesSpacing[d];
        imageBufferOrigin[d] = imageSeriesOrigin[d];
    }

    auto sliceImageBuffer = NewImage2dImage<TImageSliceType>(imageBufferSize[0], imageBufferSize[1]);
    sliceImageBuffer->SetSpacing(imageBufferSpacing);
    sliceImageBuffer->SetOrigin(imageBufferOrigin);

    typedef itk::RecursiveMultiResolutionPyramidImageFilter<TImageSliceType, TImageSliceType> PyramidImageFilterType;

    auto imagePyramide = PyramidImageFilterType::New();
    imagePyramide->SetInput(sliceImageBuffer);
    imagePyramide->SetNumberOfLevels(numberOfScales);
    imagePyramide->Update();

    // create empty image pyramides
    std::vector<typename TImageType::SizeType> imageSizes;
    std::vector<typename TImageType::Pointer> images;
    for(int i = 0; i < numberOfScales; ++i)
    {
        auto image = imagePyramide->GetOutput(i);
        auto imageSizeTMP = image->GetLargestPossibleRegion().GetSize();
        typename TImageType::SizeType sizeTMP;

        for(int d = 0; d < IMAGE_DIMENSIONS - 1; ++d)
            sizeTMP[d] = imageSizeTMP[d];

        sizeTMP[IMAGE_DIMENSIONS - 1] = imageSeriesSize[IMAGE_DIMENSIONS - 1];

        imageSizes.push_back(sizeTMP);

        auto imagePyramideTMP = NewImage3dImage<TImageType>(imageSizes[i]);
        typename TImageType::SpacingType spacingTMP;
        typename TImageType::PointType originTMP;

        for(int d = 0; d < IMAGE_DIMENSIONS - 1; ++d)
        {
            spacingTMP[d] = image->GetSpacing()[d];
            originTMP[d] = image->GetOrigin()[d];
        }

        spacingTMP[IMAGE_DIMENSIONS - 1] = imageSeriesSpacing[IMAGE_DIMENSIONS - 1];
        originTMP[IMAGE_DIMENSIONS - 1] = imageSeriesOrigin[IMAGE_DIMENSIONS - 1];

        imagePyramideTMP->SetSpacing(spacingTMP);
        imagePyramideTMP->SetOrigin(originTMP);
        images.push_back(imagePyramideTMP);
    }

    int numberOfPixelPerImage = imageSeriesSize[0]*imageSeriesSize[1];

    // create image pyramide for all images
    for(int i = 0; i < imageSeriesSize[2]; ++i)
    {

        auto sliceImageTMP = NewImage2dImage<TImageSliceType>(imageSeriesSize[0], imageSeriesSize[1]);
        sliceImageTMP->SetSpacing(imageBufferSpacing);
        sliceImageTMP->SetOrigin(imageBufferOrigin);

        std::memcpy(sliceImageTMP->GetBufferPointer(), &imageSeries->GetBufferPointer()[i*numberOfPixelPerImage],
                numberOfPixelPerImage*sizeof(ScalarType));


        auto imagePyramideSlice = PyramidImageFilterType::New();
        imagePyramideSlice->SetInput(sliceImageTMP);
        imagePyramideSlice->SetNumberOfLevels(numberOfScales);
        imagePyramideSlice->Update();

        //atach image to pyrmide syries
        for(int scaleIndex = 0; scaleIndex < numberOfScales; ++scaleIndex)
        {
            auto imageSize = imageSizes[scaleIndex];
            int numberOfPixels = imageSize[0]*imageSize[1];
            auto imageTMP = imagePyramideSlice->GetOutput(scaleIndex);
            std::memcpy(&images[scaleIndex]->GetBufferPointer()[i*numberOfPixels], imageTMP->GetBufferPointer(),
                    numberOfPixels*sizeof(ScalarType));
        }
    }
    // set the last pyramide level to the original data
    images[numberOfScales - 1] = imageSeries;

    return images;
}


template<typename TImageType, typename TDisplacementType>
typename TDisplacementType::Pointer CreateDisplacementForImage(typename TImageType::Pointer image)
{
    typename TDisplacementType::Pointer displacement = TDisplacementType::New();

    typename TDisplacementType::RegionType region;
    typename TDisplacementType::IndexType start;
    start.Fill(0);

    region.SetSize(image->GetLargestPossibleRegion().GetSize());
    region.SetIndex(start);

    typename TDisplacementType::PixelType pixel;
    pixel.Fill(0);

    displacement->SetRegions(region);
    displacement->Allocate();
    displacement->FillBuffer(pixel);

    return displacement;
}

template<typename TDisplacementType>
typename TDisplacementType::Pointer CreateDisplacement(typename TDisplacementType::SizeType size)
{
    typename TDisplacementType::Pointer displacement = TDisplacementType::New();

    typename TDisplacementType::RegionType region;
    typename TDisplacementType::IndexType start;
    start.Fill(0);

    region.SetSize(size);
    region.SetIndex(start);

    typename TDisplacementType::PixelType pixel;
    pixel.Fill(0);

    displacement->SetRegions(region);
    displacement->Allocate();
    displacement->FillBuffer(pixel);

    return displacement;
}

template<typename TImageType>
typename TImageType::Pointer CreateImage(typename TImageType::SizeType size)
{
    typename TImageType::RegionType region;
    typename TImageType::IndexType start;
    start.Fill(0);

    region.SetSize(size);
    region.SetIndex(start);

    auto image = TImageType::New();
    image->SetRegions(region);
    image->Allocate();
    image->FillBuffer(0);

    return image;
}


template<typename TImageType>
typename TImageType::Pointer NewImage2dImage(int width, int height)
{
    typename TImageType::RegionType region;
    typename TImageType::IndexType start;
    start[0] = 0;
    start[1] = 0;

    typename TImageType::SizeType size;
    size[0] = width;
    size[1] = height;

    region.SetSize(size);
    region.SetIndex(start);

    typename TImageType::Pointer image = TImageType::New();
    image->SetRegions(region);
    image->Allocate();

    typename itk::ImageRegionIterator<TImageType> iterator(image, image->GetLargestPossibleRegion());

    while(!iterator.IsAtEnd())
    {
        iterator.Set(0.0);

        ++iterator;
    }

    return image;
}

template<typename TImageType>
void ScalaDisplacement(typename TImageType::Pointer image, ScalarType value)
{
    typename itk::ImageRegionIterator<TImageType> iterator(image, image->GetLargestPossibleRegion());

    while(!iterator.IsAtEnd())
    {
        typename TImageType::PixelType pixel = iterator.Get();
        for(int d = 0; d < IMAGE_DIMENSIONS; ++d)
            pixel[d] *= value;

        iterator.Set(pixel);

        ++iterator;
    }
}

template<typename TImageType>
void ConvertToParameters(typename TImageType::Pointer image, ScalarType* parameter)
{
    typename itk::ImageRegionIterator<TImageType> iterator(image, image->GetLargestPossibleRegion());

    typename TImageType::SizeType imageSize = image->GetLargestPossibleRegion().GetSize();

    int index = 0;
    int offset = imageSize[0]*imageSize[1];

    while(!iterator.IsAtEnd())
    {
        typename TImageType::PixelType pixel = iterator.Get();
        parameter[index] = pixel[0];
        parameter[index + offset] = pixel[1];

        index++;
        ++iterator;
    }
}

template<typename TImageType>
ScalarType GetMaxValue(typename TImageType::Pointer image)
{
    typedef itk::StatisticsImageFilter<TImageType> FilterType;
    auto filter = FilterType::New();

    filter->SetInput(image);
    filter->Update();

    return filter->GetMaximum();
}

template<typename TImageType>
ScalarType GetMinValue(typename TImageType::Pointer image)
{
    typedef itk::StatisticsImageFilter<TImageType> FilterType;
    auto filter = FilterType::New();

    filter->SetInput(image);
    filter->Update();

    return filter->GetMinimum();
}

template<typename TImageType>
typename TImageType::Pointer CreateDisplacement(ScalarType *buffer, int width, int height)
{
    typename TImageType::RegionType region;
    typename TImageType::IndexType start;
    start[0] = 0;
    start[1] = 0;

    typename TImageType::SizeType size;
    size[0] = width;
    size[1] = height;

    region.SetSize(size);
    region.SetIndex(start);

    typename TImageType::Pointer displacement = TImageType::New();
    displacement->SetRegions(region);
    displacement->Allocate();

    int offset = width*height;

    for(int y = 0; y < height; ++y)
    {
        for(int x = 0; x < width; ++x)
        {
            typename TImageType::PixelType pixel;
            typename TImageType::IndexType index;
            index[0] = x;
            index[1] = y;

            for(int d = 0; d < IMAGE_DIMENSIONS; ++d)
            {
                pixel[d] = buffer[x + y*width + d*offset];

            }

            displacement->SetPixel(index, pixel);
        }
    }

    return displacement;
}


template<typename TImageType>
void SetImageParameterToStd(typename TImageType::Pointer image)
{
    // set direction to identity
    typename TImageType::DirectionType direction;
    direction.SetIdentity();
    image->SetDirection(direction);

    // set origin to zero
    typename TImageType::PointType itkOrigin;
    itkOrigin.Fill(0.0);
    image->SetOrigin(itkOrigin);

    // set spacing to 1 in all dimensions
    typename TImageType::SpacingType itkSpacing;
    itkSpacing.Fill(1.0);
    image->SetSpacing(itkSpacing);

}

template<typename TImageTypeIn, typename TImageTypeOut>
typename TImageTypeOut::Pointer calculateMeanTensor(typename TImageTypeIn::Pointer tensorImage)
{
    typename TImageTypeIn::SizeType size = tensorImage->GetLargestPossibleRegion().GetSize();

    typename TImageTypeOut::RegionType region;
    typename TImageTypeOut::IndexType start;
    start.Fill(0);

    typename TImageTypeOut::SizeType sizeOut;
    sizeOut[0] = size[0];
    sizeOut[1] = size[1];

    typename TImageTypeOut::SpacingType spacing;
    spacing[0] = tensorImage->GetSpacing()[0];
    spacing[1] = tensorImage->GetSpacing()[1];

    region.SetSize(sizeOut);
    region.SetIndex(start);

    typename TImageTypeOut::Pointer meanTeansorImage = TImageTypeOut::New();
    meanTeansorImage->SetRegions(region);
    meanTeansorImage->SetSpacing(spacing);
    meanTeansorImage->Allocate();

    for(int x = 0; x < size[1]; ++x)
    {
        for(int y = 0; y < size[1]; ++y)
        {
            typename TImageTypeOut::PixelType meanTensor;
            meanTensor(0, 0) = 0;
            meanTensor(0, 1) = 0;
            meanTensor(1, 0) = 0;
            meanTensor(1, 1) = 0;

            for(int t = 0; t < size[2]; ++t)
            {
                typename TImageTypeIn::IndexType indexIn;
                indexIn[0] = x;
                indexIn[1] = y;
                indexIn[2] = t;

                auto tensor = tensorImage->GetPixel(indexIn);

                meanTensor(0, 0) += tensor(0, 0);
                meanTensor(0, 1) += tensor(0, 1);
                meanTensor(1, 0) += tensor(1, 0);
                meanTensor(1, 1) += tensor(1, 1);

            }

            typename TImageTypeOut::IndexType indexOut;
            indexOut[0] = x;
            indexOut[1] = y;
            meanTeansorImage->SetPixel(indexOut, meanTensor);
        }
    }
    return meanTeansorImage;
}

template<typename TImageType>
typename TImageType::Pointer ShuffleImageData(typename TImageType::Pointer inputImage, std::vector<int> index)
{
    typename TImageType::Pointer outputImage = TImageType::New();
    outputImage->SetRegions(inputImage->GetRequestedRegion());
    outputImage->CopyInformation(inputImage);
    outputImage->Allocate();

    typename TImageType::SizeType size = inputImage->GetLargestPossibleRegion().GetSize();
    

#pragma omp parallel for
    for(int z = 0; z < size[2]; ++z)
    {
        for(int y = 0; y < size[1]; ++y)
        {
            
            for(int x = 0; x < size[0]; ++x)
            {
                typename TImageType::IndexType indexFrom;
                typename TImageType::IndexType indexTo;

                indexFrom[0] = x;
                indexFrom[1] = y;
                indexFrom[2] = z;

                indexTo[0] = x;
                indexTo[1] = y;
                indexTo[2] = index[z];

                outputImage->SetPixel(indexTo, inputImage->GetPixel(indexFrom));
            }
        }
    }
    return outputImage;
}

template<typename TImageType>
typename TImageType::Pointer
RescaleImage(typename TImageType::Pointer image, double min, double max)
{
    typedef itk::RescaleIntensityImageFilter<TImageType, TImageType> FilterType;
    auto filter = FilterType::New();
    filter->SetInput(image);
    filter->SetOutputMinimum(min);
    filter->SetOutputMaximum(max);
    filter->Update();
    return filter->GetOutput();
}

template<typename TImageType>
typename TImageType::Pointer
ShrinkImage(typename TImageType::Pointer image, int factorX, int factorY)
{
    typedef itk::ShrinkImageFilter<TImageType, TImageType> FilterType;
    auto filter = FilterType::New();
    filter->SetInput(image);
    filter->SetShrinkFactor(0, factorX); // shrink the first dimension by a factor of 2
    filter->SetShrinkFactor(1, factorY); // shrink the second dimension by a factor of 2
    filter->Update();
    return filter->GetOutput();
}

template<typename TImageType>
typename TImageType::Pointer
InvertImage(typename TImageType::Pointer image)
{
    typedef itk::StatisticsImageFilter<TImageType> StatisticsFilterType;
    auto statisticsImageFilter = StatisticsFilterType::New();
    statisticsImageFilter->SetInput(image);
    statisticsImageFilter->Update();

    typedef itk::InvertIntensityImageFilter<TImageType> FilterType;
    auto filter = FilterType::New();
    filter->SetInput(image);
    filter->SetMaximum(statisticsImageFilter->GetMaximum());
    filter->Update();
    return filter->GetOutput();
}



// Templated itk ReadImage function.
// Attention: if nifti files are readed, the intend_code in the header
// has to be correct (1007) for displacement fields.
template<typename TImageType>
typename TImageType::Pointer
ReadImage(const std::string& filename)
{
    typedef itk::ImageFileReader<TImageType> ReaderType;
    typename ReaderType::Pointer reader = ReaderType::New();

    reader->SetFileName(filename);
    reader->Update();

    return reader->GetOutput();
}

// Templated itk WriteImage function.
// Attention: if nifti files are readed, the intend_code in the header
// has to be correct (1007) for displacement fields.
template<typename TImageType>
void WriteImage(typename TImageType::Pointer image, const std::string& filename)
{
    typedef itk::ImageFileWriter<TImageType> WriterType;
    typename WriterType::Pointer writer = WriterType::New();

    writer->SetFileName(filename);
    writer->SetInput(image);
    writer->Update();
}

//template<typename TImageType>
//typename TImageType::Pointer
//ReadPNGSeries(const std::string& filename, const std::string& filePrefix, const int startIndex, const int endIndex)
//{
//    typedef itk::ImageSeriesReader<TImageType>  SeriesReaderType;

//    typedef itk::NumericSeriesFileNames NamesGeneratorType;
//    NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();
//    nameGenerator->SetStartIndex( startIndex );
//    nameGenerator->SetEndIndex( endIndex );
//    nameGenerator->SetIncrementIndex( 1 );
//    nameGenerator->SetSeriesFormat( filename + "/" + filePrefix );

//    typename SeriesReaderType::Pointer reader = SeriesReaderType::New();
//    reader->SetImageIO( itk::PNGImageIO::New() );
//    reader->SetFileNames( nameGenerator->GetFileNames() );
//    reader->Update();

//    return reader->GetOutput();
//}

template<typename TImageType>
typename TImageType::Pointer
ReadImageSeries(const std::string& filename, const std::string& filePrefix, const std::string& format,
                const int startIndex, const int endIndex)
{
    typedef itk::ImageSeriesReader<TImageType> SeriesReaderType;

    typedef itk::NumericSeriesFileNames NamesGeneratorType;
    NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();
    nameGenerator->SetStartIndex( startIndex );
    nameGenerator->SetEndIndex( endIndex );
    nameGenerator->SetIncrementIndex( 1 );
    nameGenerator->SetSeriesFormat( filename + "/" + filePrefix + "." + format);

    typename SeriesReaderType::Pointer reader = SeriesReaderType::New();
    if (format.compare("png") == 0)
    {
        reader->SetImageIO( itk::PNGImageIO::New() );
    }
    else if (format.compare("vtk") == 0)
    {
         reader->SetImageIO( itk::VTKImageIO::New() );
    }
    reader->SetFileNames( nameGenerator->GetFileNames() );
    reader->Update();

    return reader->GetOutput();
}

template< typename TImageType, typename TOutputImageType>
void SliceImageData(typename TImageType::Pointer imageData,  const std::string& filePrefix)
{

    typename TImageType::SizeType size = imageData->GetLargestPossibleRegion().GetSize();

    typedef itk::NumericSeriesFileNames NamesGeneratorType;

    auto nameGenerator = NamesGeneratorType::New();
    nameGenerator->SetIncrementIndex( 1 );
    nameGenerator->SetStartIndex( 0 );
    nameGenerator->SetEndIndex( size[2] - 1 );
    nameGenerator->SetSeriesFormat( filePrefix );

    typedef itk::ImageSeriesWriter< TImageType, TOutputImageType> ImageSeriesWriterType;
    auto imageSeriesWriter = ImageSeriesWriterType::New();
    imageSeriesWriter->SetInput( imageData );
    imageSeriesWriter->SetFileNames( nameGenerator->GetFileNames() );
    imageSeriesWriter->Update();
}

template<typename TTransformType, typename TImageType, typename TDisplacementFieldType>
typename TDisplacementFieldType::Pointer GenerateDisplacementField(typename TTransformType::Pointer transform,
                                                                   typename TImageType::Pointer reference_image){
    typedef typename itk::TransformToDisplacementFieldFilter<TDisplacementFieldType> DisplacementFieldGeneratorType;
    typename DisplacementFieldGeneratorType::Pointer dispfieldGenerator = DisplacementFieldGeneratorType::New();
    dispfieldGenerator->UseReferenceImageOn();
    dispfieldGenerator->SetReferenceImage(reference_image);
    //    dispfieldGenerator->SetOutputOrigin(reference_image->GetOrigin());
    //    dispfieldGenerator->SetOutputSpacing(reference_image->GetSpaceing());
    //    dispfieldGenerator->
    dispfieldGenerator->SetTransform(transform);
    dispfieldGenerator->Update();
    return dispfieldGenerator->GetOutput();
}

template<typename TImageType>
typename TImageType::Pointer
CropImage(typename TImageType::Pointer image, int startX, int startY, int lengthX, int lengthY)
{
    typedef itk::RegionOfInterestImageFilter< TImageType, TImageType> RegionOfInterestFilterType;
    auto regionFilter = RegionOfInterestFilterType::New();

    typename TImageType::SizeType sizeImage = image->GetLargestPossibleRegion().GetSize();
    typename TImageType::IndexType start;
    start[0] = startX;
    start[1] = startY;
    start[2] = 0;

    typename TImageType::SizeType size;
    size[0] = lengthX;
    size[1] = lengthY;
    size[2] = sizeImage[2];

    typename TImageType::RegionType desiredRegion;
    desiredRegion.SetSize(size);
    desiredRegion.SetIndex(start);
    regionFilter->SetRegionOfInterest(desiredRegion);
    regionFilter->SetInput(image);
    regionFilter->Update();

    return regionFilter->GetOutput();
}

template<typename TImageTypeOut, typename TImageTypeIn>
typename TImageTypeOut::Pointer
Create3dImageMaskFrom2dImage(typename TImageTypeIn::Pointer image, int numberOfImages)
{

    typedef itk::JoinSeriesImageFilter<TImageTypeIn, TImageTypeOut> FilterType;
    auto filter = FilterType::New();
    filter->SetSpacing(1.0);
    filter->SetOrigin(0.0);

    for(int i = 0; i < numberOfImages; ++i)
        filter->SetInput(i, image);
    
    
    filter->Update();

    return filter->GetOutput();
} 


template<typename TImageType>
typename TImageType::Pointer
GaussianSmoothing(typename TImageType::Pointer image, ScalarType sigma)
{
    typedef itk::DiscreteGaussianImageFilter<TImageType, TImageType> GuassianFilterType;
    auto filter = GuassianFilterType::New();
    filter->SetInput(image);
    filter->SetVariance(sigma);
    filter->SetUseImageSpacing(true);
    filter->SetMaximumKernelWidth(64);
    filter->Update();

    return filter->GetOutput();
}

template<typename TImageType>
typename TImageType::Pointer
ImageVariance(typename TImageType::Pointer image, ScalarType sigma, typename TImageType::Pointer mean)
{

    typedef itk::SquareImageFilter<TImageType, TImageType> SquareImageFilterType;
    auto squareImageFilterImage = SquareImageFilterType::New();
    squareImageFilterImage->SetInput(image);
    squareImageFilterImage->Update();

    auto squareImageFilterMean = SquareImageFilterType::New();
    squareImageFilterMean->SetInput(mean);
    squareImageFilterMean->Update();



    typedef itk::DiscreteGaussianImageFilter<TImageType, TImageType> GuassianFilterType;
    auto filter = GuassianFilterType::New();
    filter->SetInput(squareImageFilterImage->GetOutput());
    filter->SetVariance(sigma);
    filter->SetUseImageSpacing(true);
    filter->SetMaximumKernelWidth(64);
    filter->Update();

    typedef itk::SubtractImageFilter<TImageType, TImageType> SubtractImageFilterType;
    auto subtractFilter = SubtractImageFilterType::New ();
    subtractFilter->SetInput1(filter->GetOutput());
    subtractFilter->SetInput2(squareImageFilterMean->GetOutput());
    subtractFilter->Update();

    return subtractFilter->GetOutput();
}


template<typename TImageType>
typename TImageType::Pointer
MultiplyImages(typename TImageType::Pointer image1, typename TImageType::Pointer image2){

    typedef typename itk::MultiplyImageFilter<TImageType, TImageType, TImageType> MultiplyImageFilterType;
    auto multiplier = MultiplyImageFilterType::New();
    multiplier->SetInput1(image1);
    multiplier->SetInput2(image2);
    multiplier->Update();

    return multiplier->GetOutput();
}

template<typename TImageType>
typename TImageType::Pointer
SubtractImages(typename TImageType::Pointer image1, typename TImageType::Pointer image2){

    typedef itk::SubtractImageFilter<TImageType, TImageType> SubtractImageFilterType;
    auto subtractFilter = SubtractImageFilterType::New ();
    subtractFilter->SetInput1(image1);
    subtractFilter->SetInput2(image2);
    subtractFilter->Update();

    return subtractFilter->GetOutput();
}




template<typename TImageType>
typename TImageType::Pointer
MultiplyConstant(typename TImageType::Pointer image, typename TImageType::ValueType constant){

    typedef typename itk::MultiplyImageFilter<TImageType, TImageType, TImageType> MultiplyImageFilterType;
    typename MultiplyImageFilterType::Pointer multiplier = MultiplyImageFilterType::New();
    multiplier->SetInput(image);
    multiplier->SetConstant(constant);
    multiplier->Update();

    return multiplier->GetOutput();
}

template<typename TImageType>
typename TImageType::Pointer
ThresholdImage(typename TImageType::Pointer image, double thresh_max, double thresh_min)
{
    // estimate maximum below threshold
    double max = std::numeric_limits<double>::lowest();
    double min = std::numeric_limits<double>::max();
    typename itk::ImageRegionConstIterator<TImageType> iterator(image,image->GetLargestPossibleRegion());
    while(!iterator.IsAtEnd()){
        double value = static_cast<double>(iterator.Get());
        if(max<value && value<thresh_max){
            max=value;
        }
        if(min>value && value>thresh_min){
            min=value;
        }
        ++iterator;
    }

    // get minimum value
    typedef typename itk::StatisticsImageFilter<TImageType> StatisticsImageFilterType;
    typename StatisticsImageFilterType::Pointer statistics = StatisticsImageFilterType::New();
    statistics->SetInput(image);
    statistics->Update();

    typename TImageType::Pointer result;
    {
        // threshold image for max threshold
        typedef typename itk::ThresholdImageFilter<TImageType> ThresholdImageFilterType;
        typename ThresholdImageFilterType::Pointer thresher = ThresholdImageFilterType::New();
        thresher->SetInput(image);
        thresher->ThresholdAbove(thresh_max);
        thresher->ThresholdBelow(statistics->GetMinimum());
        thresher->ThresholdOutside(statistics->GetMinimum(), max);
        thresher->SetOutsideValue(max);
        thresher->Update();
        result = thresher->GetOutput();
    }

    {
        // threshold image for min threshold
        typedef typename itk::ThresholdImageFilter<TImageType> ThresholdImageFilterType;
        typename ThresholdImageFilterType::Pointer thresher = ThresholdImageFilterType::New();
        thresher->SetInput(result);
        thresher->ThresholdAbove(statistics->GetMaximum());
        thresher->ThresholdBelow(thresh_min);
        thresher->ThresholdOutside(min, statistics->GetMaximum());
        thresher->SetOutsideValue(min);
        thresher->Update();
        result = thresher->GetOutput();
    }

    return result;
}

template<typename TImageType>
typename TImageType::Pointer
MedianFilterImage(typename TImageType::Pointer image, int radius){
    if(radius<=0){
        return image;
    }

    typedef typename itk::MedianImageFilter<TImageType, TImageType> MedianImageFilterType;
    typename MedianImageFilterType::Pointer medianer = MedianImageFilterType::New();
    medianer->SetInput(image);
    medianer->SetRadius(radius);
    medianer->Update();
    return medianer->GetOutput();
}

template<typename TImageType>
typename TImageType::Pointer
HistogramMatching(typename TImageType::Pointer image, const typename TImageType::Pointer reference, unsigned levels=1024, unsigned matchpoints=7){
    // normalizes the grayscale values of a source image based on the grayscale values of a reference image
    typedef typename itk::HistogramMatchingImageFilter<TImageType, TImageType> HistogramMatchingImageFilterType;
    typename HistogramMatchingImageFilterType::Pointer hist_matcher = HistogramMatchingImageFilterType::New();
    hist_matcher->SetSourceImage(image);
    hist_matcher->SetReferenceImage(reference);
    hist_matcher->SetNumberOfHistogramLevels(levels);
    hist_matcher->SetNumberOfMatchPoints(matchpoints);
    hist_matcher->ThresholdAtMeanIntensityOff();
    hist_matcher->Update();
    return hist_matcher->GetOutput();
}

template<typename TImageType>
double AccumulateImage( typename TImageType::Pointer image){
    double accu = 0;
    typename itk::ImageRegionConstIterator<TImageType> iterator(image, image->GetLargestPossibleRegion());
    while(!iterator.IsAtEnd()){
        accu += iterator.Get();
        ++iterator;
    }
    return accu;
}

template<typename TImageType, typename TDisplacementFieldType>
typename TImageType::Pointer
WarpImage(typename TImageType::Pointer image, const typename TDisplacementFieldType::Pointer df, unsigned order=3){
    typedef typename itk::BSplineInterpolateImageFunction<TImageType, double> InterpolatorType;
    typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
    interpolator->SetSplineOrder(order);

    typedef typename itk::WarpImageFilter<TImageType, TImageType, TDisplacementFieldType> WarpImageFilterType;
    typename WarpImageFilterType::Pointer warper = WarpImageFilterType::New();
    warper->SetInterpolator(interpolator);
    warper->SetDisplacementField(df);
    warper->SetOutputParametersFromImage(df);
    warper->SetInput(image);
    warper->Update();

    return warper->GetOutput();
}

template<typename TImageType, typename TDisplacementFieldType>
typename TImageType::Pointer
WarpImage2(typename TImageType::Pointer image, typename TImageType::Pointer target, const typename TDisplacementFieldType::Pointer df, unsigned order=3){
    typedef typename itk::BSplineInterpolateImageFunction<TImageType, double> InterpolatorType;
    typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
    interpolator->SetSplineOrder(order);

    const double padding_value = std::numeric_limits<double>::lowest();

    typedef typename itk::WarpImageFilter<TImageType, TImageType, TDisplacementFieldType> WarpImageFilterType;
    typename WarpImageFilterType::Pointer warper = WarpImageFilterType::New();
    warper->SetInterpolator(interpolator);
    warper->SetDisplacementField(df);
    warper->SetOutputParametersFromImage(df);
    warper->SetInput(image);
    warper->SetEdgePaddingValue(padding_value);
    warper->Update();

    typename TImageType::Pointer warped_image = warper->GetOutput();

    if(warped_image->GetLargestPossibleRegion().GetNumberOfPixels() !=
            target->GetLargestPossibleRegion().GetNumberOfPixels()){
        return warped_image;
    }


    /// set values which have been set to the EdgePaddingValue
    /// to the corresponding target image value
    typename itk::ImageRegionConstIterator<TImageType> tar_iter(target, target->GetLargestPossibleRegion());
    typename itk::ImageRegionIterator<TImageType> war_iter(warped_image, warped_image->GetLargestPossibleRegion());
    while(!tar_iter.IsAtEnd() && !war_iter.IsAtEnd()){

        if(war_iter.Get() == padding_value){
            war_iter.Set(tar_iter.Get());
        }

        ++tar_iter;
        ++war_iter;
    }


    return warped_image;
}


template<typename TImageType>
typename TImageType::Pointer
AverageNeighborhood(typename TImageType::Pointer image, typename TImageType::SizeValueType radius){
    typedef typename itk::MeanImageFilter<TImageType,TImageType> MeanImageFilterType;
    typename MeanImageFilterType::Pointer filter = MeanImageFilterType::New();
    filter->SetInput(image);
    filter->SetRadius(radius);
    filter->Update();
    return filter->GetOutput();
}

template<typename TImageType, typename TDisplacementFieldType>
typename TImageType::Pointer
MagnitudesOfDisplacements(typename TDisplacementFieldType::Pointer df){
    typedef typename itk::VectorMagnitudeImageFilter<TDisplacementFieldType, TImageType> MagnitudeImageFilterType;
    typename MagnitudeImageFilterType::Pointer filter = MagnitudeImageFilterType::New();
    filter->SetInput(df);
    filter->Update();
    return filter->GetOutput();
}













