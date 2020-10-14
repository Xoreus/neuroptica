#!/bin/bash                                                                 
for FILE in ./*.pdf; do
    pdf2ps "${FILE}" file.ps && ps2pdf -dPDFSETTINGS="/preprint" file.ps "${FILE}"
done


