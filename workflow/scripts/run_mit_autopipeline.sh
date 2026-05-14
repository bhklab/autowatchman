#!/bin/bash

imgtools autopipeline \
--modalities 'CT,RTSTRUCT' \
-rmap "ROI:ROI-1" \
-rmap "ROI:Lymph Node.*" \
--existing-file-mode skip \
data/rawdata/PMCC_AutoWATChmAN/images/AutoWATChmAN \
data/procdata/PMCC_AutoWATChmAN/images/mit_AutoWATChmAN 