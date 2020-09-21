#!/usr/bin/env python

from __future__ import print_function

import collections
import csv
import logging
import os
import SimpleITK as sitk

import radiomics
from radiomics import featureextractor

import datetime


def main():
    ###给出基本路径
    params_path = r'E:\Radiomics\huaxi_jiang_yinjie\feature'
    base_outpath = r'E:\Radiomics\huaxi_jiang_yinjie\feature'
    base_dicom_nii_path = r'G:\West China Hospotal-Gastric Cancer SRCC\NRRD'
    series = 'pre' ##给定序列号 AP pre PVP

    outPath = os.path.join(base_outpath, series)
    dicom_nii_path = os.path.join(base_dicom_nii_path, series)###原始图片转换成nrrd文件

    ###给出输入输出文件路径
    inputCSV = os.path.join(outPath, 'presample_nii_path.csv')###文件中是分割文件的路径 'AP PVP pre'
    outputFilepath = os.path.join(outPath, 'radiomics_SRCC_pre_Primary.csv')  ###记得更改不同的特征名称 'AP PVP pre'

    progress_filename = os.path.join(outPath, 'pyrad_log.txt')
    # params = os.path.join(outPath, 'exampleSettings', 'Params.yaml')
    params = os.path.join(params_path, 'exampleCT.yaml')


    # Configure logging
    rLogger = logging.getLogger('radiomics')

    # Set logging level
    # rLogger.setLevel(logging.INFO)  # Not needed, default log level of logger is INFO

    # Create handler for writing to log file
    handler = logging.FileHandler(filename=progress_filename, mode='w')
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    rLogger.addHandler(handler)

    # Initialize logging for batch log messages
    logger = rLogger.getChild('batch')

    # Set verbosity level for output to stderr (default level = WARNING)
    radiomics.setVerbosity(logging.INFO)

    logger.info('pyradiomics version: %s', radiomics.__version__)
    logger.info('Loading CSV')

    flists = []
    try:
      with open(inputCSV, 'r') as inFile:
        cr = csv.DictReader(inFile, lineterminator='\n')
        flists = [row for row in cr]
    except Exception:
      logger.error('CSV READ FAILED', exc_info=True)

    logger.info('Loading Done')
    logger.info('Patients: %d', len(flists))

    if os.path.isfile(params):
      extractor = featureextractor.RadiomicsFeatureExtractor(params)
    else:  # Parameter file not found, use hardcoded settings instead
      settings = {}
      settings['binWidth'] = 25
      settings['resampledPixelSpacing'] = None  # [3,3,3]
      settings['interpolator'] = sitk.sitkBSpline
      settings['enableCExtensions'] = True

      extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
      # extractor.enableInputImages(wavelet= {'level': 2})

    logger.info('Enabled input images types: %s', extractor.enabledImagetypes)
    logger.info('Enabled features: %s', extractor.enabledFeatures)
    logger.info('Current settings: %s', extractor.settings)

    headers = None

    for idx, entry in enumerate(flists, start=1):
      newname = entry['name'] + '.nrrd'#####改名字
      # logger.info("(%d/%d) Processing Patient (Image: %s, Mask: %s)", idx, len(flists), entry['Image'], entry['Mask'])
      logger.info("(%d/%d) Processing Patient (Image: %s, Mask: %s)", idx, len(flists), os.path.join(dicom_nii_path, newname), entry['path'])###更改mask_path和primary名称

      # imageFilepath = entry['Image']
      # maskFilepath = entry['Mask']

      imageFilepath = os.path.join(dicom_nii_path, newname)
      # maskFilepath = entry['primary']###更改原发灶和淋巴结
      maskFilepath = entry['path']  ###更改原发灶和淋巴结
      label = entry.get('Label', None)

      if str(label).isdigit():
        label = int(label)
      else:
        label = None

      if (imageFilepath is not None) and (maskFilepath is not None):
        featureVector = collections.OrderedDict(entry)
        featureVector['Image'] = os.path.basename(imageFilepath)
        featureVector['Mask'] = os.path.basename(maskFilepath)

        try:
          featureVector.update(extractor.execute(imageFilepath, maskFilepath, label))

          with open(outputFilepath, 'a') as outputFile:
            writer = csv.writer(outputFile, lineterminator='\n')
            if headers is None:
              headers = list(featureVector.keys())
              writer.writerow(headers)

            row = []
            for h in headers:
              row.append(featureVector.get(h, "N/A"))
            writer.writerow(row)
        except Exception:
          logger.error('FEATURE EXTRACTION FAILED', exc_info=True)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print(f'开始时间：{start_time}, 结束时间：{end_time}, 历时：{(end_time - start_time).seconds}')

